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
epochs = 5
lr=5e-4
num_freqs=61
random.seed(42)

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

# Specify CSV file path
csv_path_geometries = data_dir + "/KITqm9_geometries.csv"
geometries = ut.load_geometry(csv_path_geometries)

# Print some sample molecules
for key, value in list(geometries.items())[:3]:  # Print first 3 molecules
    print(f"IDX: {key}, Data: {value}")

# Example: Accessing a molecule's data
idx_to_check = 34  # Example index
if idx_to_check in geometries:
    molecule = geometries[idx_to_check]
    print(f"\nMolecule {idx_to_check}:")
    print("Atomic Numbers (z):", molecule.z)
    print("Geometries (pos):", molecule.pos)
else:
    print(f"Molecule {idx_to_check} not found in dataset.")

dataset = []
frequencies = ut.load_unique_frequencies(csv_path)

if not frequencies:
    print("No valid frequency found in CSV.")

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
        if idx in geometries:
            mol = geometries[idx]
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
            idx = mol.idx,
            pos=pos.to(torch.float32),    # Atomic positions
            z=torch.LongTensor(z),        # Atomic numbers
            freq=torch.tensor(float(freq_val), dtype=torch.float32),
            y=y,  # Polarizability tensor (target)
        )
        if spectrum_value < 0.000005:#  high_spec_cutoff:
            dataset.append(data_entry)
            count += 1
        else:
            pass
            # Randomly sample ~0.2% of the "low-spec" data
            #if random.random() < low_fraction:
             #   dataset.append(data_entry)
                


print(f"Collected {count} high-spec (>0.1) entries.")
print(f"Total dataset length: {len(dataset)}")

ex1 = dataset[0]
ex2 = dataset[5]

print("dataset[0] :", ex1.idx, ex1.freq, ex1.spec)
print("dataset[5] :", ex2.idx, ex2.freq, ex2.spec)

spec_values = [item.spec.item() for item in dataset]
spec_mean = np.mean(spec_values)
spec_std = np.std(spec_values)

print("Spec mean, std =", spec_mean, spec_std)
for item in dataset:
    old_val = item.spec.item()
    norm_val = (old_val - spec_mean) / (spec_std + 1e-8)  # avoid div by zero
    item.spec = torch.tensor(norm_val, dtype=torch.float32)


# NORMALIZATION OF POLAR_VALUES => COMBINED
"""
import numpy as np

y_vals = []

for item in dataset:
    y = item.y.reshape(-1).tolist()   
    y_vals.extend(y)
# compute mean, std
y_mean, y_std = np.mean(y_vals), np.std(y_vals)

print("y mean, std =", y_mean, y_std)

# Now transform each data entry
for item in dataset:
    y = item.y  # [3,6]
    # 4) do standard z-score
    y_norm = (y - y_mean)/(y_std + 1e-8)
    item.y = y_norm
"""
    

# NORMALIZATION OF POLAR_VALUES => SEPAPRATE
"""
import numpy as np

real_vals = []
imag_vals = []

for item in dataset:
    y = item.y  # shape [3,6]
    # real => columns [:,:3]
    # imag => columns [:,3:]
    real_part = y[:, :3].reshape(-1).tolist()   # shape [9]
    imag_part = y[:, 3:].reshape(-1).tolist()   # shape [9]
    real_vals.extend(real_part)
    imag_vals.extend(imag_part)

# compute mean, std
real_mean, real_std = np.mean(real_vals), np.std(real_vals)
imag_mean, imag_std = np.mean(imag_vals), np.std(imag_vals)

print("Real mean, std =", real_mean, real_std)
print("Imag mean, std =", imag_mean, imag_std)

# Now transform each data entry
for item in dataset:
    y = item.y  # [3,6]
    
    # real => y[:, :3], shape [3,3]
    # imag => y[:, 3:], shape [3,3]
    real_slice = y[:, :3]
    imag_slice = y[:, 3:]
    
    # 4) do standard z-score
    real_norm = (real_slice - real_mean)/(real_std + 1e-8)
    imag_norm = (imag_slice - imag_mean)/(imag_std + 1e-8)
    
    # reassign
    y[:, :3] = real_norm
    y[:, 3:] = imag_norm
    
    item.y = y
"""


# MIN MAX normalizaiton
"""
# 1) First pass: find global min & max of spec
spec_list = []
for item in dataset:
    spec_list.append(item.spec.item())

spec_min = min(spec_list)
spec_max = max(spec_list)
print("spec min, max =", spec_min, spec_max)

for item in dataset:
    old_val = item.spec.item()
    norm_val = (old_val - spec_min) / (spec_max - spec_min)
    print(norm_val)
    item.spec = torch.tensor(norm_val, dtype=torch.float32)


# Suppose each data_entry.y is shape [3,6].
# real part: y[:,:3], imag part: y[:,3:]

real_vals = []
imag_vals = []

for item in dataset:
    y = item.y  # shape [3,6]
    # Flatten each part
    real_part = y[:, :3].reshape(-1)  # shape [9], since 3x3
    imag_part = y[:, 3:].reshape(-1)  # shape [9], since 3x3

    real_vals.extend(real_part.tolist())
    imag_vals.extend(imag_part.tolist())

real_min, real_max = min(real_vals), max(real_vals)
imag_min, imag_max = min(imag_vals), max(imag_vals)

print("Real part range:", real_min, real_max)
print("Imag part range:", imag_min, imag_max)


for item in dataset:
    y = item.y  # shape [3,6]
    # real => columns [:,:3]
    # imag => columns [:,3:]

    # 2a) Real part
    # shape [3,3]
    real_slice = y[:, :3]
    real_norm = (real_slice - real_min) / (real_max - real_min)
    y[:, :3] = real_norm

    # 2b) Imag part
    imag_slice = y[:, 3:]
    imag_norm = (imag_slice - imag_min) / (imag_max - imag_min)
    y[:, 3:] = imag_norm

    item.y = y
"""


# train and validate per frequencies
"""
num_val_freqs = max(1, int(0.1 * len(frequencies)))  # Ensure at least 1 frequency is selected
print("len frequencies ", len(frequencies))
print("num_val_freqs", num_val_freqs)

val_frequencies = set(random.sample(frequencies, num_val_freqs))
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
"""


# Train and validate per molecule
unique_mol_ids = list({data_entry.idx for data_entry in dataset})
random.shuffle(unique_mol_ids)

num_val_mols = max(1, int(0.2 * len(unique_mol_ids)))  # e.g., 10% for validation
val_mol_ids = set(unique_mol_ids[:num_val_mols])

train_datasets = [d for d in dataset if d.idx not in val_mol_ids]
val_datasets = [d for d in dataset if d.idx in val_mol_ids]

print(f"Total unique molecules: {len(unique_mol_ids)}")
print(f"Validation molecule IDs: {sorted(val_mol_ids)}")
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

torch.save(model.state_dict(), current_dir + f'/trained_param/ee_polarizabilities_all_freq_KITqm9_smaller_than_0.000005_no_normalization.pth')