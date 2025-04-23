import copy
import os
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

from pathlib import Path
import trainer
import json 

from detanet_model import *
import wandb

batch_size = 8
epochs = 100
lr=5e-4


def load_dataset(dataset, geometries):
    new_dataset = []
    print(dataset[0])
    KIT_mol = []
    for molecule in dataset:
        if molecule.number in geometries:
            KIT_mol = geometries[molecule.number]
            print("KIT_mol number ", KIT_mol.idx)
            print("molecule.number ", molecule.number)
        else:
            continue 
    
        z_old = molecule['z']
        #print("qm9s z", z_old)

        #
        z = KIT_mol.z
        #print("KIT_geo_z", z)

        # pos = KIT_mol.pos
        pos = molecule['pos']
        polar = molecule['polar'].squeeze()

        complex_polar = torch.zeros(3,3)
        y = torch.concat([polar, complex_polar], dim=-1)
        
        # Create the dataset entry
        data_entry = Data(
            idx=KIT_mol.idx,
            pos=pos.to(torch.float32),    # Atomic positions
            z=torch.LongTensor(z),        # Atomic numbers
            y=y      # polar.to(torch.float32)  # Polarizability tensor (target)
        )
        new_dataset.append(data_entry)
    return new_dataset


def load_dataset_from_csv(csv_filename="new_dataset.csv"):
    """
    Reads the dataset from a CSV file produced by save_dataset_to_csv,
    then reconstructs the Data objects.
    """
    df = pd.read_csv(csv_filename)
    loaded_dataset = []

    for _, row in df.iterrows():
        # Convert stringified lists back to Python lists, then Tensors
        pos_list = json.loads(row["pos"])
        z_list = json.loads(row["z"])
        y_list = json.loads(row["y"])

        pos_tensor = torch.tensor(pos_list, dtype=torch.float32)
        z_tensor   = torch.tensor(z_list,  dtype=torch.long)
        y_tensor   = torch.tensor(y_list,  dtype=torch.float32)

        data_entry = Data(pos=pos_tensor, z=z_tensor, y=y_tensor)
        loaded_dataset.append(data_entry)

    return loaded_dataset


# Utility function to convert atomic symbols to atomic numbers
def element_to_atomic_number(element):
    periodic_table = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18
        # Add more elements if necessary
    }
    return periodic_table.get(element, 0)  # Default to 0 if element is unknown

# Load the dataset from the CSV file
def load_geometry(csv_path_geometries):
    dataset_dict = {}  # Dictionary with idx as key

    with open(csv_path_geometries, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        header = next(csv_reader)  # Read header
        print("header", header)
        # Column indices
        idx_col = header.index("idx")
        num_molecules_col = header.index("Number of Molecules")
        atoms_col = header.index("Atoms")
        geometries_col = header.index("Geometries")

        for row in csv_reader:
            try:
                idx = int(row[idx_col])  # Get molecule ID
            except ValueError:
                print(f"Skipping invalid idx: {row[idx_col]}")
                continue

            try:
                # Parse atomic symbols and convert to atomic numbers
                atom_symbols = json.loads(row[atoms_col].replace("'", "\""))  # Convert single quotes to double for JSON
                atomic_numbers = [element_to_atomic_number(el) for el in atom_symbols]
                z = torch.tensor(atomic_numbers, dtype=torch.long)

                # Parse geometry (3D coordinates)
                geometries = json.loads(row[geometries_col])  # Expected format: [[x, y, z], ...]
                pos = torch.tensor(geometries, dtype=torch.float32)

                # Create a PyTorch Geometric Data object
                data_entry = Data(
                    idx=idx,
                    z=z,       # Atomic numbers
                    pos=pos    # Atomic positions
                )
                # Store in dictionary
                dataset_dict[idx] = data_entry


            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing row for idx {idx}: {e}")
                continue

    return dataset_dict


current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, 'data')

# Specify CSV file path
csv_path_geometries = data_dir + "/KITqm9_geometries.csv"
geometries = load_geometry(csv_path_geometries)

# Print some sample molecules
for key, value in list(geometries.items())[:3]:  # Print first 3 molecules
    print(f"IDX: {key}, Data: {value}")

# Example: Accessing a molecule's data
idx_to_check = 34  # Example index
if idx_to_check in geometries:
    molecule = geometries[idx_to_check]
    print(f"\nMolecule {idx_to_check}:")
    print("Atomic Numbers (z):", molecule.z)
    print("Geometries (pos):", molecule.pos)
else:
    print(f"Molecule {idx_to_check} not found in dataset.")


logging.basicConfig(
    filename=parent_dir + "/log/train_detanet.log", # '/pfs/work7/workspace/scratch/pf1892-ws/logs/training_detaNet.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info(f"torch.cuda.is_available() {torch.cuda.is_available()}")


qm9s = torch.load( data_dir +"/qm9s.pt")
dataset = load_dataset(qm9s, geometries)
# dataset = load_dataset_from_csv(data_dir + "/QM9s_160.csv")

print("length dataset ", len(dataset))
print(dataset[0])

print("Length of dataset: ", len(dataset))
ex1 = dataset[0]
ex2 = dataset[5]

print("dataset[0] :", ex1.idx, ex1.y)
print("dataset[5] :", ex2.idx, ex2.y)


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

wandb.init(
    # set the wandb project where this run will be logged
    project="Detanet",
    name=f"qm9s_only_zero_freq",
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "GNN",
    "dataset": "QM9s",
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
                    scalar_outsize= 4, # 2,#4, 
                    irreps_out= '2x2e', #'2e',# '2e+2e',
                    summation=True,
                    norm=False,
                    out_type='complex_2_tensor', # '2_tensor',
                    grad_type=None,
                    device=device)
model.train()
model.to(device) 
wandb.watch(model, log="all")

'''Finally, using the trainer, training 20 times from a 5e-4 learning rate'''
trainer=trainer.Trainer(model,train_loader=trainloader,val_loader=valloader,loss_function=ut.fun_complex_mse_loss,lr=lr,weight_decay=0,optimizer='AdamW')
trainer.train(num_train=epochs,targ='y')

torch.save(model.state_dict(), current_dir + '/trained_param/ee_polarizabilities.pth')