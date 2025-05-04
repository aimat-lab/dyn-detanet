
import copy
import os.path as osp
import csv
import util.utils as ut
import pandas as pd
import torch
from torch.nn.functional import one_hot
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import global_mean_pool
import torch_geometric
import logging
import os
from pathlib import Path
import json 

from detanet_model import *
from data_processing.preprocess_data import load_polarizabilities, save_dataset_to_csv

import wandb

batch_size = 64
epochs = 100
lr=5e-4

wandb.init(
    # set the wandb project where this run will be logged
    project="dynDetanet",

    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "GNN",
    "dataset": "HarvardOPV",
    "epochs": epochs,
    }
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = DynDetaNet(num_features=128,
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
                    scalar_outsize=2, # 2,#4, 
                    irreps_out='2e',# '2e+2e',
                    summation=True,
                    norm=False,
                    out_type='2_tensor', # '2_tensor',
                    grad_type=None,
                    device=device)
model.train()
model.to(device) 
wandb.watch(model, log="all")

'''Next, define the trainer and the parameters used for training.'''
class Trainer:
    def __init__(self,model,train_loader,val_loader=None,loss_function=l2loss,device=torch.device(device),
                 optimizer='Adam_amsgrad',lr=5e-4,weight_decay=0):
        self.opt_type=optimizer
        self.device=device
        self.model=model
        self.train_data=train_loader
        self.val_data=val_loader
        self.device=device
        self.opts={'AdamW':torch.optim.AdamW(self.model.parameters(),lr=lr,amsgrad=False,weight_decay=weight_decay),
              'AdamW_amsgrad':torch.optim.AdamW(self.model.parameters(),lr=lr,amsgrad=True,weight_decay=weight_decay),
              'Adam':torch.optim.Adam(self.model.parameters(),lr=lr,amsgrad=False,weight_decay=weight_decay),
              'Adam_amsgrad':torch.optim.Adam(self.model.parameters(),lr=lr,amsgrad=True,weight_decay=weight_decay),
              'Adadelta':torch.optim.Adadelta(self.model.parameters(),lr=lr,weight_decay=weight_decay),
              'RMSprop':torch.optim.RMSprop(self.model.parameters(),lr=lr,weight_decay=weight_decay),
              'SGD':torch.optim.SGD(self.model.parameters(),lr=lr,weight_decay=weight_decay)
        }
        self.optimizer=self.opts[self.opt_type]
        self.loss_function=loss_function
        self.step=-1
    def train(self,num_train,targ,stop_loss=1e-8, val_per_train=50, print_per_epoch=10):
        self.model.train()
        len_train=len(self.train_data)
        for i in range(num_train):
            val_datas=iter(self.val_data)
            for j,batch in enumerate(self.train_data):
                self.step=self.step+1
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()
                out = self.model(pos=batch.pos.to(self.device), z=batch.z.to(self.device), batch=batch.batch.to(self.device))
                #print("out", out)
                target = batch[targ].to(self.device)

                #print("target," , target)
                loss = self.loss_function(out.reshape(target.shape),target)
                loss.backward()
                self.optimizer.step()
                wandb.log({"train_loss": loss, "epoch": i})
                if (self.step%val_per_train==0) and (self.val_data is not None):
                    val_batch = next(val_datas)
                    val_out = self.model(pos=val_batch.pos.to(self.device), z=val_batch.z.to(self.device), batch=val_batch.batch.to(self.device))
                    val_target=val_batch[targ].to(self.device).reshape(-1)

                    # Ensure the shapes match
                    val_loss = self.loss_function(val_out.reshape(val_target.shape), val_target.to(self.device)).item()
                    val_mae = l1loss(val_out.reshape(val_target.shape), val_target.to(self.device)).item()
                    val_R2 = R2(val_out.reshape(val_target.shape), val_target.to(self.device)).item()
                    wandb.log({"val_loss": val_loss, "epoch": i})
                    wandb.log({"val_mae": val_mae, "epoch": i})
                    wandb.log({"val_R2": val_R2, "epoch": i})

                    if self.step % print_per_epoch==0:
                        logging.info('Epoch[{}/{}],loss:{:.8f},val_loss:{:.8f},val_mae:{:.8f},val_R2:{:.8f}'
                              .format(self.step,num_train*len_train,loss.item(),val_loss,val_mae,val_R2))

                    assert (loss > stop_loss) or (val_loss > stop_loss),'Training and prediction Loss is less' \
                                                                        ' than cut-off Loss, so training stops'
                elif (self.step % print_per_epoch == 0) and (self.step%val_per_train!=0):
                    logging.info('Epoch[{}/{}],loss:{:.8f}'.format(self.step,num_train*len_train, loss.item()))
                    
    def load_state_and_optimizer(self,state_path=None,optimizer_path=None):
        if state_path is not None:
            state_dict=torch.load(state_path)
            self.model.load_state_dict(state_dict)
        if optimizer_path is not None:
            self.optimizer=torch.load(optimizer_path)

    def save_param(self,path):
        torch.save(self.model.state_dict(),path)

    def save_model(self,path):
        torch.save(self.model, path)

    def save_opt(self,path):
        torch.save(self.optimizer,path)

    def params(self):
        return self.model.state_dict()


current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, 'data')

logging.basicConfig(
    filename=parent_dir + "/log/train_dyn_detanet.log", # '/pfs/work7/workspace/scratch/pf1892-ws/logs/training_detaNet.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info(f"torch.cuda.is_available() {torch.cuda.is_available()}")


# Function to generate one-hot encodings
def generate_onehot(z, max_atomic_number):
    indices = z - 1  # Shift atomic numbers to 0-based index
    return one_hot(indices, num_classes=max_atomic_number).float()

def get_dataset(dataset):
    # Maximum atomic number in the dataset
    max_atomic_number = 9

    new_dataset = []

    for molecule in dataset:
        pos = molecule['pos']
        z = molecule['z']
        polar = molecule['polar'].squeeze()

        # Create the dataset entry
        data_entry = torch_geometric.data.Data(
            pos=pos.to(torch.float32),    # Atomic positions
            z=torch.LongTensor(z),        # Atomic numbers
            y=torch.tensor(polar, dtype=torch.float32)        # polar.to(torch.float32)  # Polarizability tensor (target)
        )

        new_dataset.append(data_entry)
    
    return new_dataset


dataset = torch.load("../data/qm9s.pt")
print(dataset[0])
dataset = get_dataset(dataset)

train_datasets=[]
val_datasets=[]
for i in range(len(dataset)):
    if i%10==0:
        val_datasets.append(dataset[i])
    else:
        train_datasets.append(dataset[i])

'''Using torch_Geometric.dataloader.DataLoader Converts a dataset into a batch of 64 molecules of training data.'''

trainloader=DataLoader(train_datasets,batch_size=batch_size,shuffle=True)
valloader=DataLoader(val_datasets,batch_size=batch_size,shuffle=True)

'''Finally, using the trainer, training 20 times from a 5e-4 learning rate'''
trainer=Trainer(model,train_loader=trainloader,val_loader=valloader,loss_function=l2loss,lr=lr,weight_decay=0,optimizer='AdamW')
trainer.train(num_train=epochs,targ='y')

torch.save(model.state_dict(), current_dir + '/trained_param/qm9s_polar.pth')
