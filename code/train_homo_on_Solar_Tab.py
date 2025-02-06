import copy
import os.path as osp
import csv

import utils as ut

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

from detanet_model import *

logging.basicConfig(
    filename='/pfs/work7/workspace/scratch/pf1892-ws/logs/training_detaNet.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info(f"torch.cuda.is_available() {torch.cuda.is_available()}")


# File path (replace with the actual file path)
csv_path = "/media/maria/work_space/capsule-3259363/data/HarvardOPV_revised.csv" #data/HarvardOPV_40.csv"

# Open the CSV file and process it
dataset = []
with open(csv_path, newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    
    # Read the header to identify column indices
    header = next(csv_reader)
    smiles_idx = header.index("SMILES")
    homo_idx = header.index("HOMO(eV)")
    lumo_idx = header.index("LUMO(eV)")
    
    # Read each row
    for row in csv_reader:
        smiles = row[smiles_idx]
        homo = torch.tensor([float(row[homo_idx])], dtype=torch.float) 
        lumo = torch.tensor([float(row[lumo_idx])], dtype=torch.float32)
        # Convert SMILES to molecular graph
        graph_data = ut.smiles_to_graph(smiles)
        if graph_data is None:
            continue  # Skip invalid molecules
        
        z, pos = graph_data
        
        # Create a PyTorch Geometric Data object
        data_entry = Data(
            pos=pos.to(torch.float32),    # Atomic positions
            z=torch.LongTensor(z),        # Atomic numbers
            y=torch.tensor([homo, lumo], dtype=torch.float32),  # Polarizability tensor (target)
        )
        
        dataset.append(data_entry)


train_datasets=[]
val_datasets=[]
for i in range(len(dataset)):
    if i%10==0:
        val_datasets.append(dataset[i])
    else:
        train_datasets.append(dataset[i])
        
len(train_datasets),len(val_datasets)


'''Using torch_Geometric.dataloader.DataLoader Converts a dataset into a batch of 64 molecules of training data.'''
batches=16
trainloader=DataLoader(train_datasets,batch_size=batches,shuffle=True)
valloader=DataLoader(val_datasets,batch_size=batches,shuffle=True)


device = torch.device("cpu")

'''After loading the dataset, we train a model using NPA charge as an example.
 	Firstly, construct an untrained model:'''
model=DetaNet(num_features=128,
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
                 scale=None,
                 scalar_outsize=2,
                 irreps_out=None,
                 summation=False,
                 norm=False,
                 out_type='scalar',
                 grad_type=None ,
                 device=device)

model.train()



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
                out = self.model(pos=batch.pos.to(self.device), z=batch.z.to(self.device),
                                     batch=batch.batch.to(self.device))
                #print("out", out.shape)
                graph_out = global_mean_pool(out, batch.batch)  # Shape: [batch_size, d]

                # print("graph_out", graph_out.shape)

                target = batch[targ].to(self.device)

                #print("target" , target.shape)
                loss = self.loss_function(graph_out.reshape(target.shape),target)
                loss.backward()
                self.optimizer.step()
                if (self.step%val_per_train==0) and (self.val_data is not None):
                    val_batch = next(val_datas)
                    val_target=val_batch[targ].to(self.device).reshape(-1)

                    val_out = self.model(pos=val_batch.pos.to(self.device), z=val_batch.z.to(self.device),
                    batch=val_batch.batch.to(self.device))
                    # Aggregate node-level outputs to graph-level outputs
                    val_graph_out = global_mean_pool(val_out, val_batch.batch)  # Shape: [val_batch_size, d]

                    # Ensure the shapes match
                    val_loss = self.loss_function(val_graph_out.reshape(val_target.shape), val_target).item()
                    val_mae = l1loss(val_graph_out.reshape(val_target.shape), val_target).item()
                    val_R2 = R2(val_graph_out.reshape(val_target.shape), val_target).item()

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
    



'''Finally, using the trainer, training 20 times from a 5e-4 learning rate'''
trainer=Trainer(model,train_loader=trainloader,val_loader=valloader,loss_function=l2loss,lr=5e-4,weight_decay=0,optimizer='AdamW')


trainer.train(num_train=25,targ='y')

torch.save(model.state_dict(),'trained_param/homo_lumo_HarvardOPV.pth')

eval_loader = DataLoader(val_datasets, batch_size=1, shuffle=False)

import matplotlib.pyplot as plt
import torch
from torch_geometric.nn import global_mean_pool

# Get predictions
predictions = []
true_values = []

for batch in eval_loader:
    true_values.append(batch.y.unsqueeze(0))  # Ensure correct shape
    with torch.no_grad():
        val_out = model(pos=batch.pos.to(device), z=batch.z.to(device),
                                batch=batch.batch.to(device))
        val_graph_out = global_mean_pool(val_out, batch.batch)  # Shape: [batch_size, d]
        predictions.append(val_graph_out)

# Convert lists of tensors to a single tensor
true_values = torch.cat(true_values, dim=0).cpu()  # Now shape [num_samples, 2]
predictions = torch.cat(predictions, dim=0).cpu()  # Now shape [num_samples, 2]

# Separate HOMO and LUMO values
true_homo_values = true_values[:, 0]  # Convert to eV
true_lumo_values = true_values[:, 1]  # Convert to eV
predictions_homo_values = predictions[:, 0] 
predictions_lumo_values = predictions[:, 1] 

# 🔵 Plot HOMO results
plt.figure(figsize=(8, 6))
plt.scatter(true_homo_values, predictions_homo_values, c='blue', alpha=0.7, label='Predictions')
plt.plot([min(true_homo_values), max(true_homo_values)], [min(true_homo_values), max(true_homo_values)], color='red', linestyle='--', label='y=x')
plt.xlabel('True HOMO Values (eV)')
plt.ylabel('Predicted HOMO Values (eV)')
plt.title('HOMO Predictions vs True Values')
plt.legend()
plt.grid(True)
plt.savefig("SolarTab_homo_prediction.png")

# 🟢 Plot LUMO results
plt.figure(figsize=(8, 6))
plt.scatter(true_lumo_values, predictions_lumo_values, c='green', alpha=0.7, label='Predictions')
plt.plot([min(true_lumo_values), max(true_lumo_values)], [min(true_lumo_values), max(true_lumo_values)], color='red', linestyle='--', label='y=x')
plt.xlabel('True LUMO Values (eV)')
plt.ylabel('Predicted LUMO Values (eV)')
plt.title('LUMO Predictions vs True Values')
plt.legend()
plt.grid(True)
plt.savefig("SolarTab_lumo_prediction.png")
