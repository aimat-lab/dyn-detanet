
import os
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DataLoader

import trainer_per_elem_normalized as trainer_per_elem_normalized
from detanet_model import *

from sklearn.preprocessing import StandardScaler
import numpy as np
import torch


import wandb
import random
seed = 0
random.seed(seed)

batch_size = 32
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
dataset_name = 'KITQM9'
x_features = 61

name = f"per_elem_clipped_{x_features}xfeatures{epochs}epochs_{batch_size}batchsize_{lr}lr_{cutoff}cutoff_{num_block}numblock_{num_features}features_{dataset_name}_seed{seed}"



import torch

def print_stats(datasets, label=""):
    """datasets – list of PyG Data objects with .real_ee and .imag_ee"""
    real_flat = torch.cat([d.real_ee.flatten() for d in datasets])
    imag_flat = torch.cat([d.imag_ee.flatten() for d in datasets])

    for part, tensor in [("real", real_flat), ("imag", imag_flat)]:
        print(f"{label:>6} | {part:4} | "
              f"mean = {tensor.mean():10.4f}   "
              f"var  = {tensor.var(unbiased=False):10.4f}   "
              f"min  = {tensor.min():10.4f}   "
              f"max  = {tensor.max():10.4f}")



def clip_by_value(matrix_2d: torch.Tensor,
                  low:  float = -2_500.0,
                  high: float =  2_500.0) -> torch.Tensor:
    """
    Return a copy of `matrix_2d` (shape [N, 9]) where every entry
    outside the interval [low, high] is set to the nearest bound.
    """
    return torch.clamp(matrix_2d, min=low, max=high)




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

# Subtract static polarizability
for data in dataset:
    data.real_ee = data.real_ee[1:]-data.real_ee[0]
    data.imag_ee = data.imag_ee[1:]

    data.y = torch.cat([data.real_ee, data.imag_ee], dim=0)

    if x_features == 61:
        data.x = data.spectra[1:].repeat(len(data.z), 1)
    elif x_features == 80:
        x = torch.cat([data.osc_pos, data.osc_strength], dim= 0)
        data.x = x.repeat(len(data.z), 1)


print_stats(dataset, "Before normalization (whole dataset)")

# Split the dataset into training and validation sets
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


# Normalize the data (standard z-score)
real_rows = torch.cat([d.real_ee.reshape(-1, 9) for d in train_datasets])  # [N·61, 9]
imag_rows = torch.cat([d.imag_ee.reshape(-1, 9) for d in train_datasets])  # [N·61, 9]


print("real_rows shape:", real_rows.shape)  # [N·61, 9]
print("imag_rows shape:", imag_rows.shape)  # [N·61, 9]


#robust_real = StandardScaler()     
#R_scaled = robust_real.fit_transform(real_rows.numpy()).astype("float32")

#robust_imag = StandardScaler()
#I_scaled = robust_imag.fit_transform(imag_rows.numpy()).astype("float32")


real_rows_clipped = clip_by_value(real_rows, -2_500, 2_500)
imag_rows_clipped = clip_by_value(imag_rows, -2_500, 2_500)

print("real_rows_clipped shape:", real_rows_clipped.shape)  # [N·61, 9]
print("imag_rows_clipped shape:", imag_rows_clipped.shape)  # [N·61, 9]

# 2) standard-scale the clipped data
std_real = StandardScaler()
std_imag = StandardScaler()

#R_clip = std_real.fit_transform(real_rows_clipped.numpy()).astype("float32")
#I_clip = std_imag.fit_transform(imag_rows_clipped.numpy()).astype("float32")

std_real.fit(real_rows_clipped.numpy())
std_imag.fit(imag_rows_clipped.numpy())

# Apply the scaling to **original (unclipped) data**
R_clip = ((real_rows.numpy() - std_real.mean_) / std_real.scale_).astype("float32")
I_clip = ((imag_rows.numpy() - std_imag.mean_) / std_imag.scale_).astype("float32")

offset = 0
for d in train_datasets:
    n = d.real_ee.shape[0]                                # 61 frequency slices
    d.real_ee = torch.tensor(R_clip[offset:offset+n]).view(n, 3, 3)
    d.imag_ee = torch.tensor(I_clip[offset:offset+n]).view(n, 3, 3)
    offset += n
    d.y = torch.cat([d.real_ee, d.imag_ee], dim=0)  # [122, 3, 3]

print_stats(train_datasets, "After normalization (training set)")

# Clip the validation set
for mol in val_datasets:
    # flatten the data
    real_val_flat = mol.real_ee.reshape(-1, 9)
    imag_val_flat = mol.imag_ee.reshape(-1, 9)

    # clip with training thresholds
    real_val_clip = clip_by_value(real_val_flat, -2_500, 2_500)
    imag_val_clip = clip_by_value(imag_val_flat, -2_500, 2_500)

    # write back
    n = mol.real_ee.shape[0]
    mol.real_ee = real_val_clip.view(n, 3, 3)
    mol.imag_ee = imag_val_clip.view(n, 3, 3)
    mol.y       = torch.cat([mol.real_ee, mol.imag_ee], dim=0)


# Dataloaders
trainloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True,  drop_last=False)
valloader   = DataLoader(val_datasets,   batch_size=batch_size, shuffle=False, drop_last=False)


wandb.init(
    # set the wandb project where this run will be logged
    project="OPT-configs",
    name=name,
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "GNN",
    "dataset": dataset_name,
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


trainer_ = trainer_per_elem_normalized.Trainer(
    model,
    train_loader=trainloader,
    val_loader=valloader,
    loss_function=l2loss, #ut.fun_complex_multidimensional_loss, 
    lr=lr,
    weight_decay=0,
    optimizer='AdamW'
)

trainer_.train(num_train=epochs, targ=target, real_scaler=std_real, imag_scaler=std_imag)

torch.save(model.state_dict(), os.path.join(current_dir, 'trained_param', name + '.pth'))
