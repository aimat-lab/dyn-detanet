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
epochs = 550
lr=5e-4

normalize = False
fine_tune = False #True 

len_freq_centers = 0 # To be initialialized later 

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, 'data')

dataset = []

dataset = torch.load(os.path.join(data_dir, 'HOPV_KITqm9_dataset.pt'))
print(dataset[0])
for data in dataset:
    data.y = torch.cat([data.osc_pos, data.osc_strength], dim=0)


random.shuffle(dataset)
train_frac = 0.9
split_index = int(train_frac * len(dataset))

train_datasets = dataset[:split_index]
val_datasets   = dataset[split_index:]

#print(f"Validation molecule IDs: {sorted(val_mol_ids)}")
print(f"Training set size: {len(train_datasets)}")
print(f"Validation set size: {len(val_datasets)}")


'''Using torch_Geometric.dataloader.DataLoader Converts a dataset into a batch of 64 molecules of training data.'''

trainloader=DataLoader(train_datasets,batch_size=batch_size,shuffle=True, drop_last=True)
valloader=DataLoader(val_datasets,batch_size=batch_size,shuffle=True, drop_last=True)


name = f"uv-spectra-15-osc_pos_mol34_lr_{lr}_epochs_{epochs}"
wandb.init(
    # set the wandb project where this run will be logged
    project="Detanet-uv-spec",
    name=name, 
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "GNN",
    "dataset": "KITQM9",
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
                    scalar_outsize=15*2, 
                    irreps_out= None,
                    summation=True,
                    norm=False,
                    out_type='scalar', 
                    grad_type=None,
                    device=device)

if fine_tune:
    state_dict = torch.load("code/trained_param/uv-spectra-mal_mol34-len_61-batch_64-lr_0.0005-epochs_80.pth")
    model.load_state_dict(state_dict=state_dict)
model.train()
model.to(device)
wandb.watch(model, log="all")


'''Finally, using the trainer, training 20 times from a 5e-4 learning rate'''
trainer_=trainer.Trainer(model,train_loader=trainloader,val_loader=valloader,loss_function=l2loss ,lr=lr,weight_decay=0,optimizer='AdamW')
trainer_.train(num_train=epochs,targ='y')

torch.save(model.state_dict(), current_dir + "/"+ name + ".pth")
