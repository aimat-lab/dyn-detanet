import copy
import os
import os.path as osp
import csv
import utils as ut
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
import matplotlib.pyplot as plt

from pathlib import Path
import trainer
import json 

from detanet_model import *
import wandb
import random
random.seed(42)

batch_size = 64
epochs = 80
lr=5e-4

normalize = False
fine_tune = False # Need to train on the appropriate number of samples

len_freq_centers = 0 # To be initialialized later 

##dataset = load_dataset(csv_path=csv_path, qm9_path=qm9_path)

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, 'data')
csv_path = data_dir + "/uv_boraden.csv"


qm9s_path = os.path.join(data_dir, "qm9s.pt")
qm9s = torch.load(qm9s_path)
print("Loaded qm9s dataset.")
print("length of qm9s ", len(qm9s))

count = 0
dataset = []

with open(csv_path, newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    
    # Read the header to identify column indices
    frequencies = next(csv_reader)
    frequencies = frequencies[1:]


    # Read each row
    for row in csv_reader:
        try:
            idx = int(row[0])
        except ValueError:
            print("Can't read index:", row[0])
            continue

        freq_intensities = [
            float(x)
            for x in row[1:]
            if x.strip()  # Make sure it's not empty or whitespace
        ]

        freq_intensities = torch.tensor(freq_intensities, dtype=torch.float32)

        # 1) Define the new frequency axis with 241 points
        #freq_resampled = np.linspace(1.5, 13.5, 241)  # shape = (241,)
        freq_resampled = np.linspace(1.5, 6.5, 61)  # shape = (241,)

        # 2) Interpolate your intensities onto this new axis
        intens_resampled = np.interp(freq_resampled, frequencies, freq_intensities)

        # 3) Convert back to torch if desired
        freq_resampled_torch = torch.tensor(freq_resampled,     dtype=torch.float32)
        intens_resampled_torch = torch.tensor(intens_resampled, dtype=torch.float32)

        len_freq_centers = len(intens_resampled_torch)



        #print("freq_intensities shape", freq_intensities.shape)
        #print("length freq_intesitites", len(freq_intensities))
        
        # ----------------
        # Normalization: Scale all intensities so their max is 1
        # ----------------
        if normalize:
            max_val = intens_resampled_torch.max()
            if max_val > 0:
                intens_resampled_torch = intens_resampled_torch / max_val


        mol = qm9s[idx]
        data_entry = Data(
            idx = idx,
            number = mol.number,
            z = torch.LongTensor(mol.z),
            pos = mol.pos.to(torch.float32),
            #pos=pos.to(torch.float32),    # Atomic positions
            #z=torch.LongTensor(z),        # Atomic numbers
            y=intens_resampled_torch,  # Polarizability tensor (target)
        )
        dataset.append(data_entry)

ex1 = dataset[-1]
print(ex1.idx, ex1.number, ex1.z, ex1.pos, ex1.y)
print("Total length of dataset ", len(dataset))

# Train and validate per molecule
unique_mol_ids = list({data_entry.idx for data_entry in dataset})
random.shuffle(unique_mol_ids)

num_val_mols = max(1, int(0.1 * len(unique_mol_ids)))  # e.g., 10% for validation
val_mol_ids = set(unique_mol_ids[:num_val_mols])

train_datasets = [d for d in dataset if d.idx not in val_mol_ids]
val_datasets = [d for d in dataset if d.idx in val_mol_ids]

print(f"Total unique molecules: {len(unique_mol_ids)}")
#print(f"Validation molecule IDs: {sorted(val_mol_ids)}")
print(f"Training set size: {len(train_datasets)}")
print(f"Validation set size: {len(val_datasets)}")


'''Using torch_Geometric.dataloader.DataLoader Converts a dataset into a batch of 64 molecules of training data.'''

trainloader=DataLoader(train_datasets,batch_size=batch_size,shuffle=True, drop_last=True)
valloader=DataLoader(val_datasets,batch_size=batch_size,shuffle=True, drop_last=True)

logging.basicConfig(
    filename=parent_dir + "/log/train_detanet.log", # '/pfs/work7/workspace/scratch/pf1892-ws/logs/training_detaNet.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info(f"torch.cuda.is_available() {torch.cuda.is_available()}")

wandb.init(
    # set the wandb project where this run will be logged
    project="Detanet-uv-spec",
    name=f"uv-spectra-mse-norm_{normalize}-len_{len_freq_centers}-batch_{batch_size}-lr_{lr}-epochs_{epochs}", 
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
                    max_atomic_number=34,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=len_freq_centers,
                    irreps_out=None,
                    summation=True,
                    norm=False,
                    out_type='scalar',
                    grad_type=None,
                    device=device)

if fine_tune:
    state_dict = torch.load("/media/maria/work_space/dyn-detanet/code/trained_param/ee_polarizabilities_all_freq_KITqm9_smaller_than_0.000005_no_N_with_S.pth")
    model.load_state_dict(state_dict=state_dict)
model.train()
model.to(device)
wandb.watch(model, log="all")


'''Finally, using the trainer, training 20 times from a 5e-4 learning rate'''
trainer=trainer.Trainer(model,train_loader=trainloader,val_loader=valloader,loss_function=l2loss,lr=lr,weight_decay=0,optimizer='AdamW')
trainer.train(num_train=epochs,targ='y')

torch.save(model.state_dict(), current_dir + f'/trained_param/uv-spectra-mal_mol34-len_{len_freq_centers}-batch_{batch_size}-lr_{lr}-epochs_{epochs}.pth')
wandb.finish()

