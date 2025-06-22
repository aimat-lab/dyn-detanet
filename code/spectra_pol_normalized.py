
import os
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DataLoader

import trainer_normalized
from detanet_model import *

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch


import wandb
import random
random.seed(42)

batch_size = 8
epochs = 70
lr=0.0006
cutoff=6
num_block=6 # Try again
num_features=256
attention_head=64
num_radial=64

scalar_outsize= (4* 61)#(4*62)
irreps_out= '122x2e' #'124x1e + 124x2e'
out_type = 'multi_tensor'
target = 'y'
dataset_name = 'HOPV'
x_features = 61

name = f"normalized_{x_features}xfeatures{epochs}epochs_{batch_size}batchsize_{lr}lr_{cutoff}cutoff_{num_block}numblock_{num_features}features_{dataset_name}"


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
print("dataset[0] :", ex1, )
print("dataset[5] :", ex2,)

for data in dataset:
    data.real_ee = data.real_ee[1:]-data.real_ee[0]
    data.imag_ee = data.imag_ee[1:]
    data.y = torch.cat([data.real_ee, data.imag_ee], dim=0)  # shape: [124, 3, 3]

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
import torch
import numpy as np

all_values = []
for data in train_datasets:
    data.y = torch.cat([data.real_ee, data.imag_ee], dim=0)  # shape: [124, 3, 3]
    all_values.append(data.y.flatten().numpy())              # shape: [124*3*3] = [1116]

all_values = np.concatenate(all_values, axis=0)              # shape: [N_total_values]

global_mean = np.mean(all_values)
global_std = np.std(all_values)
print(f"Global mean: {global_mean}")
print(f"Global std: {global_std}")

for data in train_datasets:
    flat = data.y.flatten()
    norm_flat = (flat - global_mean) / global_std
    data.y = norm_flat.view(124, 3, 3).to(torch.float32)
"""



# 1) gather every raw value  →  shape [N_total_values, 1]
real_vals = torch.cat([g.real_ee.flatten() for g in train_datasets]).view(-1, 1)
imag_vals = torch.cat([g.imag_ee.flatten() for g in train_datasets]).view(-1, 1)

print("real_vals.shape :", real_vals.shape)  # shape: [N_total_values, 1]
print("imag_vals.shape :", imag_vals.shape)  # shape: [N_total_values, 1]

# 2) fit robust scalers  (e.g. keep central 96 % → (2,98) quantile range)
real_scaler = RobustScaler(quantile_range=(2.0, 98.0)).fit(real_vals.numpy())
imag_scaler = RobustScaler(quantile_range=(2.0, 98.0)).fit(imag_vals.numpy())

# 3) transform and write back ------------------------------------------------
idx = 0
for g in train_datasets:
    shape = g.real_ee.shape
    print("shape", shape)  # shape: [124, 3, 3]

    print(g.real_ee.flatten().unsqueeze(1).shape) 
    # slice this molecule’s block, reshape back to [124,3,3]
    g.real_ee = torch.from_numpy(
                   real_scaler.transform(g.real_ee.flatten().unsqueeze(1).numpy())
                 ).view_as(g.real_ee).to(torch.float32)
    g.imag_ee = torch.from_numpy(
                   imag_scaler.transform(g.imag_ee.flatten().unsqueeze(1).numpy())
                 ).view_as(g.imag_ee).to(torch.float32)

    # target tensor = real ⊕ imag  (still shape [124,3,3])
    g.y = torch.cat([g.real_ee, g.imag_ee], dim=0)


print("→ robust scaling applied: real & imag handled separately")

# 4) save the scalers for inverse transform later
torch.save({"real_scaler": real_scaler,
            "imag_scaler": imag_scaler},
           "Normalize_per_value_robust.pth")


print("2 %, 98 %⟩ quantile range used:",
      real_scaler.quantile_range)        # default (25.0, 75.0)

# Per-feature IQR (or MAD if unit_variance=True) ───────────────
print("real scale_ \n",
      real_scaler.scale_)                # shape = (9,)

print("real centre_ \n",
      real_scaler.center_)               # shape = (9,)

print("imag scale_ \n",
        imag_scaler.scale_)                # shape = (9,)
print("imag centre_ \n",
        imag_scaler.center_)               # shape = (9,)



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


trainer_ = trainer_normalized.Trainer(
    model,
    train_loader=trainloader,
    val_loader=valloader,
    loss_function=l2loss, #ut.fun_complex_multidimensional_loss, 
    lr=lr,
    weight_decay=0,
    optimizer='AdamW'
)

trainer_.train(num_train=epochs, targ=target,  real_scaler=real_scaler, imag_scaler=imag_scaler )

torch.save(model.state_dict(), os.path.join(current_dir, 'trained_param', name + '.pth'))
