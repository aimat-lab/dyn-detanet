import copy
import os
import os.path as osp
import csv
import utils as ut
import util_load_data as ud
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

batch_size = 16
epochs = 20
lr=5e-4


current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, 'data')

dataset = []
spec_data = []

# Load the dataset
dataset = torch.load(os.path.join(data_dir, 'HOPV_KITqm9_dataset.pt'))
print(f"Number of graphs in the dataset: {len(dataset)}")

print(f"Total dataset length: {len(dataset)}")

ex1 = dataset[0]
ex2 = dataset[5]

print("dataset[0] :", ex1, )
print("dataset[5] :", ex2,)

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
    item.spec = torch.tensor(norm_val, dtype=torch.float32)
"""
"""
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


# -------------------------------
# Shuffle & Train/Val Split
# -------------------------------
random.shuffle(dataset)
train_frac = 0.9
split_index = int(train_frac * len(dataset))

train_datasets = dataset[:split_index]
val_datasets   = dataset[split_index:]

print(f"Training set size: {len(train_datasets)}")
print(f"Validation set size: {len(val_datasets)}")

# Dataloaders
trainloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True,  drop_last=True)
valloader   = DataLoader(val_datasets,   batch_size=batch_size, shuffle=False, drop_last=True)


name = f"train_real_parts_{epochs}epochs_{batch_size}batchsize_{lr}lr.pt"
wandb.init(
    # set the wandb project where this run will be logged
    project="Detanet-complex",
    name=name,
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "GNN",
    "dataset": "KITqm9+HOPV",
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
                    attention_head=32,
                    rc=2.0,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=34,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=(2*62), # 2,#4, 
                    irreps_out= '62x2e', #'2e',# '2e+2e',
                    summation=True,
                    norm=False,
                    out_type='cal_multi_tensor',
                    grad_type=None,
                    device=device)

model.train()
model.to(device)
wandb.watch(model, log="all")


trainer_ = trainer.Trainer(
    model,
    train_loader=trainloader,
    val_loader=valloader,
    loss_function=l2loss, #ut.fun_complex_multidimensional_loss, 
    lr=lr,
    weight_decay=0,
    optimizer='AdamW'
)

trainer_.train(num_train=epochs, targ='imag')

torch.save(model.state_dict(), os.path.join(current_dir, 'trained_param', 'pol_spec_imag.pth'))
