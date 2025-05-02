import os
import torch
from torch_geometric.loader import DataLoader
import trainer
import wandb
import random

from detanet_model import *

# -------------------------------
# Config
# -------------------------------
random.seed(42)
batch_size = 64
epochs = 100
lr = 5e-4

normalize = False
fine_tune = False 
pol_type = 'ee'

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir   = os.path.join(parent_dir, 'data')

# Merge into a single dataset
dataset = torch.load(os.path.join(data_dir, 'HOPV_KITqm9_dataset.pt'))
print(f"Combined dataset size: {len(dataset)}")

ex1 = dataset[0]
print("ex1:", ex1)  # Only if y is defined

# -------------------------------
# Shuffle & Train/Val Split
# -------------------------------
random.shuffle(dataset)
train_frac = 0.9
split_index = int(train_frac * len(dataset))

train_datasets = dataset[:split_index]
val_datasets   = dataset[split_index:]

for mol in val_datasets:
    print("val mol:", mol.idx, mol.dataset_name)

    
print(f"Training set size: {len(train_datasets)}")
print(f"Validation set size: {len(val_datasets)}")

# Dataloaders
trainloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True,  drop_last=True)
valloader   = DataLoader(val_datasets,   batch_size=batch_size, shuffle=False, drop_last=True)

# -------------------------------
# Initialize Weights & Biases (WandB)
# -------------------------------
wandb.init(
    project="Detanet-pol-spec",
    name="HOPVKITqm9-imag",
    config={
       "learning_rate": lr,
       "architecture": "GNN",
       "dataset": "HOPV + KITQM9",
       "epochs": epochs,
    }
)

# -------------------------------
# Create Model
# -------------------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = DetaNet(
    num_features=128,
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
    scalar_outsize=(4*62),
    irreps_out='124x2e',
    summation=True,
    norm=False,
    out_type='complex_61_tensor', # e.g. your custom config
    grad_type=None,
    device=device
)
if fine_tune:
    state_dict = torch.load(os.path.join(current_dir, "trained_param", "____.pth"))
    model.load_state_dict(state_dict=state_dict)

model.to(device)
model.train()
wandb.watch(model, log="all")

# -------------------------------
# Train
# -------------------------------
trainer_ = trainer.Trainer(
    model,
    train_loader=trainloader,
    val_loader=valloader,
    loss_function=l2loss, #ut.fun_complex_multidimensional_loss, 
    lr=lr,
    weight_decay=0,
    optimizer='AdamW'
)

trainer_.train(num_train=epochs, targ='polar')

torch.save(model.state_dict(), os.path.join(current_dir, 'trained_param', 'pol_spec_imag.pth'))
