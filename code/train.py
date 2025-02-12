import copy
import os.path as osp
import csv
import utils as ut
import pandas as pd

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
import tqdm

from detanet_model import *
from preprocess_data import load_polarizabilities, save_dataset_to_csv


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
                    max_atomic_number=34,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=4, # 2,#4, 
                    irreps_out='2x2e', #'2e',# '2e+2e',
                    summation=True,
                    norm=False,
                    out_type='complex_2_tensor', # '2_tensor',
                    grad_type=None,
                    device=device)
model.train()
model.to(device) 

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
                out = self.model(pos=batch.pos.to(self.device), z=batch.z.to(self.device), freq=batch.freq.to(self.device),
                                 batch=batch.batch.to(self.device))
                target = batch[targ].to(self.device)
                loss = self.loss_function(out.reshape(target.shape),target)
                loss.backward()
                self.optimizer.step()
                if (self.step%val_per_train==0) and (self.val_data is not None):
                    val_batch = next(val_datas)
                    val_target=val_batch[targ].to(self.device).reshape(-1)

                    val_out = self.model(pos=val_batch.pos.to(self.device), z=val_batch.z.to(self.device), freq=batch.freq.to(self.device), batch=val_batch.batch.to(self.device))

                    # Ensure the shapes match
                    val_loss = self.loss_function(val_out.reshape(val_target.shape), val_target.to(self.device)).item()
                    val_mae = l1loss(val_out.reshape(val_target.shape), val_target.to(self.device)).item()
                    val_R2 = R2(val_out.reshape(val_target.shape), val_target.to(self.device)).item()

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

csv_path = data_dir + "/ee_polarizabilities.csv"

logging.basicConfig(
    filename=parent_dir + "/logging/train_dyn_detanet.log", # '/pfs/work7/workspace/scratch/pf1892-ws/logs/training_detaNet.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info(f"torch.cuda.is_available() {torch.cuda.is_available()}")

def sym3x3_to_values(mat3x3: torch.Tensor) -> torch.Tensor:
    """
    Given a 3x3 symmetric matrix (as a Tensor),
    extract the 6 unique elements in the order:
      [xx, xy, xz, yy, yz, zz]
    Returns a Tensor of shape [6].
    """
    # mat3x3 shape is [3, 3]
    # Ensure it's 2D
    xx = mat3x3[0, 0]
    xy = mat3x3[0, 1]
    xz = mat3x3[0, 2]
    yy = mat3x3[1, 1]
    yz = mat3x3[1, 2]
    zz = mat3x3[2, 2]

    return torch.stack([xx, xy, xz, yy, yz, zz])


dataset = []
with open(csv_path, newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    
    # Read the header to identify column indices
    header = next(csv_reader)
    smiles_idx = header.index("smiles")
    frequency_idx = header.index("frequency")
    matrix_real_idx = header.index("matrix_real")
    matrix_imag_idx = header.index("matrix_imag")
    
    # Read each row
    for row in csv_reader:
        smiles = row[smiles_idx]
        freq = row[frequency_idx]
        matrix_real_str = row[matrix_real_idx]
        matrix_imag_str = row[matrix_imag_idx]
        try:
            freq_val = float(freq)
        except ValueError:
            continue

        # Parse JSON for real matrix
        try:
            real_3x3 = json.loads(matrix_real_str)  # shape expected [3,3]
        except json.JSONDecodeError:
            print("Warning: Could not parse real part of matrix")
            continue

        # Parse JSON for imaginary matrix
        try:
            imag_3x3 = json.loads(matrix_imag_str)
        except json.JSONDecodeError:
            print("Warning: Could not parse imaginary part of matrix")
            continue

        # Convert to torch Tensors (shape [3,3])
        real_mat = torch.tensor(real_3x3, dtype=torch.float32)
        imag_mat = torch.tensor(imag_3x3, dtype=torch.float32)

        # Each is a 3x3 symmetric matrix; extract the 6 unique elements
        real_values = sym3x3_to_values(real_mat)  # shape [6]
        imag_values = sym3x3_to_values(imag_mat)  # shape [6]

        # Concatenate => 12 elements total
        # first 6 = real, last 6 = imaginary
        y = torch.cat([real_values, imag_values], dim=0)  # shape [12]

        # Convert SMILES to molecular graph
        graph_data = ut.smiles_to_graph(smiles)
        if graph_data is None:
            continue  # Skip invalid molecules
        
        z, pos = graph_data
        
        # Create a PyTorch Geometric Data object
        data_entry = Data(
            pos=pos.to(torch.float32),    # Atomic positions
            z=torch.LongTensor(z),        # Atomic numbers
            freq=torch.tensor(float(freq), dtype=torch.float32),
            y=y,  # Polarizability tensor (target)
        )
        
        dataset.append(data_entry)
        if len(dataset) == 143:
            break

train_datasets=[]
val_datasets=[]
for i in range(len(dataset)):
    if i%10==0:
        val_datasets.append(dataset[i])
    else:
        train_datasets.append(dataset[i])

'''Using torch_Geometric.dataloader.DataLoader Converts a dataset into a batch of 64 molecules of training data.'''
batches=16
trainloader=DataLoader(train_datasets,batch_size=batches,shuffle=True)
valloader=DataLoader(val_datasets,batch_size=batches,shuffle=True)

'''Finally, using the trainer, training 20 times from a 5e-4 learning rate'''
trainer=Trainer(model,train_loader=trainloader,val_loader=valloader,loss_function=l2loss,lr=5e-4,weight_decay=0,optimizer='AdamW')
trainer.train(num_train=50,targ='y')

torch.save(model.state_dict(), current_dir + '/trained_param/ee_polarizabilities.pth')