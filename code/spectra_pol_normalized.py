
import os
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DataLoader

import trainer_normalized
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

scalar_outsize= (4* 62)#(4*62)
irreps_out= '124x2e' #'124x1e + 124x2e'
out_type = 'multi_tensor'
target = 'y'
dataset_name = 'KITQM9'
x_features = 62 

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
    data.y = torch.cat([data.real_ee, data.imag_ee], dim=0)

    if x_features == 62:
        data.x = data.spectra.repeat(len(data.z), 1)
    elif x_features == 80:
        x = torch.cat([data.osc_pos, data.osc_strength], dim= 0)
        data.x = x.repeat(len(data.z), 1)
print("data.y.shape :", data.y.shape)


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

trainer_.train(num_train=epochs, targ=target, mean=global_mean, std=global_std)

torch.save(model.state_dict(), os.path.join(current_dir, 'trained_param', name + '.pth'))
