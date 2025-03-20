import copy
import os
import os.path as osp
import csv
import util as ut
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import global_mean_pool
import torch_geometric
import logging

from pathlib import Path
import trainer
import json 

from detanet_model import *
import wandb
import random


batch_size = 128
epochs = 5
lr=5e-5
num_freqs=61


##dataset = load_dataset(csv_path=csv_path, qm9_path=qm9_path)

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, 'data')
csv_path = data_dir + "/ee_polarizabilities_qm9s.csv"

logging.basicConfig(
    filename=parent_dir + "/log/train_detanet.log", # '/pfs/work7/workspace/scratch/pf1892-ws/logs/training_detaNet.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info(f"torch.cuda.is_available() {torch.cuda.is_available()}")

qm9s = torch.load(os.path.join(data_dir, "qm9s.pt"))
print("qm9s is loaded.")

# Build a dictionary keyed by the .number attribute once
qm9_dict = {mol.number: mol for mol in qm9s}

dataset = []
frequencies = []

with open(csv_path, newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    header = next(csv_reader)
    freq_idx = header.index("frequency")

    for row in csv_reader:
        if not row:
            continue
        try:
            f_val = float(row[freq_idx])
            frequencies.append(f_val)
        except ValueError:
            # skip invalid freq
            pass

if not frequencies:
    print("No valid frequency found in CSV.")

frequencies = list(set(frequencies)) # get unique elements by transorming into a set and back
frequencies.sort()
frequencies = frequencies[0:num_freqs]
print("reduced frequencies", frequencies)


with open(csv_path, newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    
    # Read the header to identify column indices
    header = next(csv_reader)
    frequency_idx = header.index("frequency")
    matrix_real_idx = header.index("matrix_real")
    matrix_imag_idx = header.index("matrix_imag")
    
    # Read each row
    for row in csv_reader:
        try:
            idx = int(row[0])
        except ValueError:
            print("Can't read index:", row[0])
            continue

        freq_str = row[frequency_idx]
        try:
            freq_val = float(freq_str)
        except ValueError:
            continue

        if freq_val not in frequencies:
            continue

        mol = None
        # Now you can look up any 'idx' in constant time
        if idx in qm9_dict:
            mol = qm9_dict[idx]
        else:
            continue
        
        pos = mol.pos
        z = mol.z

        # Parse JSON for real matrix
        matrix_real_str = row[matrix_real_idx]
        matrix_imag_str = row[matrix_imag_idx]
        try:
            real_3x3 = json.loads(matrix_real_str)  # expected shape [3,3]
        except json.JSONDecodeError:
            print("Warning: Could not parse real part of matrix for idx:", idx)
            continue

        try:
            imag_3x3 = json.loads(matrix_imag_str)  # expected shape [3,3]
        except json.JSONDecodeError:
            print("Warning: Could not parse imaginary part of matrix for idx:", idx)
            continue

        real_mat = torch.tensor(real_3x3, dtype=torch.float32)
        imag_mat = torch.tensor(imag_3x3, dtype=torch.float32)
        
        y = torch.cat([real_mat, imag_mat], dim=-1)  # shape [12]
            
        data_entry = Data(
            idx = mol.number,
            smiles = mol.smile,
            pos=pos.to(torch.float32),    # Atomic positions
            z=torch.LongTensor(z),        # Atomic numbers
            freq=torch.tensor(float(freq_val), dtype=torch.float32),
            y=y,  # Polarizability tensor (target)
        )
        dataset.append(data_entry)

print("Length of dataset: ", len(dataset))
ex1 = dataset[0]
ex2 = dataset[5]

print("dataset[0] :", ex1.idx, ex1.freq, ex1.y)
print("dataset[5] :", ex2.idx, ex2.freq, ex2.y)

# Select 10% of frequencies for validation
num_val_freqs = max(1, int(0.1 * len(frequencies)))  # Ensure at least 1 frequency is selected
val_frequencies = set(random.sample(frequencies, num_val_freqs))
print(f"Validation frequencies: {val_frequencies}")
val_frequencies = {float(f) for f in val_frequencies}
print(f"Validation frequencies: {val_frequencies}")

# Split dataset based on selected validation frequencies
train_datasets = []
val_datasets = []

for data_entry in dataset:
    flag = False
    for freq in val_frequencies:
        if (abs(freq - data_entry.freq.item()) < 0.0001):
            flag = True
    if flag:
        val_datasets.append(data_entry)
    else:
        train_datasets.append(data_entry)

print(f"Training set size: {len(train_datasets)}")
print(f"Validation set size: {len(val_datasets)}")

'''Using torch_Geometric.dataloader.DataLoader Converts a dataset into a batch of 64 molecules of training data.'''

trainloader=DataLoader(train_datasets,batch_size=batch_size,shuffle=True)
valloader=DataLoader(val_datasets,batch_size=batch_size,shuffle=True)



wandb.init(
    # set the wandb project where this run will be logged
    project="Detanet-freq-learn",
    name=f"Freqs[0:{num_freqs}]_bs{batch_size}", 
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "GNN",
    "dataset": "QM9s",
    "epochs": epochs,
    }
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = DetaNet(num_features=128,
                    act='swish',
                    maxl=3,
                    num_block=3,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=8,
                    rc=5.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=9,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize= 4, # 2,#4, 
                    irreps_out= '2x2e', #'2e',# '2e+2e',
                    summation=True,
                    norm=False,
                    out_type='complex_2_tensor', # '2_tensor',
                    grad_type=None,
                    device=device)
model.train()
model.to(device) 
wandb.watch(model, log="all")


'''Finally, using the trainer, training 20 times from a 5e-4 learning rate'''
trainer=trainer.Trainer(model,train_loader=trainloader,val_loader=valloader,loss_function=ut.fun_complex_mse_loss,lr=lr,weight_decay=0,optimizer='AdamW')
trainer.train(num_train=epochs,targ='y')

torch.save(model.state_dict(), current_dir + f'/trained_param/ee_polarizabilities_all_freq_KITqm9.pth')