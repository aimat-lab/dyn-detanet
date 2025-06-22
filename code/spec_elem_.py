
import os
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DataLoader

import trainer_spec_elem
from detanet_model import *

from sklearn.preprocessing import StandardScaler
import numpy as np
import torch


import wandb
import random
random.seed(42)

batch_size = 32
epochs = 70
lr=0.0006
cutoff=6
num_block=6 # Try again
num_features=256
attention_head=64
num_radial=64

scalar_outsize= 122
irreps_out= None
out_type = 'scalar'
target = 'y'
dataset_name = 'KITQM9'
x_features = 61

name = f"one_elem_{x_features}xfeatures{epochs}epochs_{batch_size}batchsize_{lr}lr_{cutoff}cutoff_{num_block}numblock_{num_features}features_{dataset_name}"


# -------------------------------

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, 'data')

dataset = []

# Load the dataset
dataset = torch.load(os.path.join(data_dir, dataset_name + '.pt'))
print(f"Number of graphs in the dataset: {len(dataset)}")

ex1 = dataset[0]
ex2 = dataset[5]
print("dataset[0] :", ex1 )
print("dataset[5] :", ex2,)


for data in dataset:
    data.real_ee = data.real_ee[1:]-data.real_ee[0]
    data.imag_ee = data.imag_ee[1:]

    data.y = torch.cat([data.real_ee, data.imag_ee], dim=0)

    if x_features == 61:
        data.x = data.spectra[1:].repeat(len(data.z), 1)
    elif x_features == 80:
        x = torch.cat([data.osc_pos, data.osc_strength], dim= 0)
        data.x = x.repeat(len(data.z), 1)


# -------------------------------
# Shuffle & Train/Val Split
# -------------------------------
random.shuffle(dataset)
train_frac = 0.9
split_index = int(train_frac * len(dataset))

train_datasets = dataset[:split_index]
val_datasets   = dataset[split_index:]

val_dataset_to_print = []
for mol in val_datasets:
    val_dataset_to_print.append(str(mol.idx))

print(f"Training set size: {len(train_datasets)}")
print(f"Validation set size: {len(val_datasets)}")


""""
from sklearn.preprocessing import StandardScaler
import torch

# Real: Build matrix of shape [N, 9]
real_rows = []
for data in train_datasets:
    # data.real : [61, 3, 3]  →  [61, 9]
    real_rows.extend(data.real.view(-1, 9))

R = torch.stack(real_rows)          # shape [N*62, 9]
print("R.shape :", R.shape)
real_scaler = StandardScaler()

R_scaled = real_scaler.fit_transform(R).astype("float32")
print("R_scaled.shape :", R_scaled.shape)


offset = 0
for data in train_datasets:
    block = R_scaled[offset : offset + 61]      # shape (62, 9)
    offset += 61
    data.real = torch.tensor(block).view(61, 3, 3)


# Imaginary: Build matrix of shape [N, 9]
imag_matrix = []
for data in train_datasets:
    imag_matrix.extend(data.imag.view(-1, 9))

I = torch.stack(imag_matrix).numpy()
imag_scaler = StandardScaler()
I_scaled = imag_scaler.fit_transform(I).astype("float32")


offset = 0
for data in train_datasets:
    block = I_scaled[offset : offset + 61]      # shape (62, 9)
    offset += 61
    data.imag = torch.tensor(block).view(61, 3, 3)

# save scaler
torch.save({'real_scaler': real_scaler,
            'imag_scaler': imag_scaler},
           'Normalize_per_element.pth')
    
for data in train_datasets:
    data.y = torch.cat([data.real, data.imag], dim=0)
"""

from sklearn.preprocessing import RobustScaler
import torch

# -----------------------------------------------------------
# Normalise **only** the (0, 0) element of each 3×3 tensor
# -----------------------------------------------------------
from sklearn.preprocessing import RobustScaler
import torch

# ----------  gather the (0,0) slices  ----------
real_00 = torch.cat([d.real_ee[:, 0, 0] for d in train_datasets]).view(-1, 1)   # [N·61, 1]
imag_00 = torch.cat([d.imag_ee[:, 0, 0] for d in train_datasets]).view(-1, 1)   # [N·61, 1]

# ----------  fit scalers on training set only  ----------
scaler_real00 = StandardScaler()
scaler_imag00 = StandardScaler()

real00_scaled = scaler_real00.fit_transform(real_00.numpy()).astype("float32").flatten()
imag00_scaled = scaler_imag00.fit_transform(imag_00.numpy()).astype("float32").flatten()

# ----------  write the scaled values back  ----------
offset = 0
for d in train_datasets:
    n = d.real_ee.shape[0]                       # 61 frequency slices
    d.real_ee = d.real_ee.clone()                # keep other components untouched
    d.imag_ee = d.imag_ee.clone()
    d.real_ee[:, 0, 0] = torch.tensor(real00_scaled[offset:offset+n])
    d.imag_ee[:, 0, 0] = torch.tensor(imag00_scaled[offset:offset+n])
    offset += n
    d.y = torch.cat([d.real_ee[:, 0, 0], d.imag_ee[:, 0, 0]], dim=0)  # 122-long target

# ----------  apply the very same scalers to validation data ----------
for d in val_datasets:
    d.real_ee = d.real_ee.clone()
    d.imag_ee = d.imag_ee.clone()
    d.y = torch.cat([d.real_ee[:, 0, 0], d.imag_ee[:, 0, 0]], dim=0)

# ----------  save the two 1-D scalers for use at test time ----------
torch.save({'real00_scaler': scaler_real00,
            'imag00_scaler': scaler_imag00},
           'Normalize_00component.pth')



# Dataloaders
trainloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True,  drop_last=True)
valloader   = DataLoader(val_datasets,   batch_size=batch_size, shuffle=False, drop_last=True)


wandb.init(
    # set the wandb project where this run will be logged
    project="normalized-spectra-input-polar",
    name=name,
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "GNN",
    "dataset": "HOPV",
    "epochs": epochs,
    }
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = DetaNet(num_features=num_features,
                    act='swish',
                    maxl=3,
                    num_block=num_block, 
                    radial_type='trainable_bessel',
                    num_radial=num_radial,
                    attention_head=attention_head,
                    rc=cutoff,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=34,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=scalar_outsize, 
                    irreps_out= irreps_out,
                    summation=True,
                    norm=False,
                    out_type=out_type,
                    grad_type=None,
                    x_features=x_features,
                    device=device)

model.train()
model.to(device)
wandb.watch(model, log="all")


trainer_ = trainer_spec_elem.Trainer(
    model,
    train_loader=trainloader,
    val_loader=valloader,
    loss_function=l2loss, #ut.fun_complex_multidimensional_loss, 
    lr=lr,
    weight_decay=0,
    optimizer='AdamW'
)

trainer_.train(num_train=epochs, targ=target, real_scaler=scaler_real00, imag_scaler=scaler_imag00)

torch.save(model.state_dict(), os.path.join(current_dir, 'trained_param', name + '.pth'))
