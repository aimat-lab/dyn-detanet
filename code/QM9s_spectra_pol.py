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
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import global_mean_pool
import torch_geometric
import matplotlib.pyplot as plt

from pathlib import Path
import trainer_pol_uv
import json 

from detanet_model import *
import wandb
import random
random.seed(42)

batch_size = 64
epochs = 100
lr=5e-4


normalize = True
fine_tune = False 
pol_type = 'ee'
lorentz = False

len_freq_centers = 0 # To be initialialized later 

##dataset = load_dataset(csv_path=csv_path, qm9_path=qm9_path)

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, 'data')

dataset = []
spec_data = []
csv_path = data_dir + "/uv_boraden.csv"

qm9s_path = os.path.join(data_dir, "qm9s.pt")
qm9s = torch.load(qm9s_path)
print("Loaded qm9s dataset.")
print("length of qm9s ", len(qm9s))


with open(csv_path, newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    
    # Read the header to identify column indices
    frequencies = next(csv_reader)
    frequencies = frequencies[1:]

    # Read each row
    for row in csv_reader:
        try:
            idx = int(row[0])
        except ValueError:
            print("Can't read index:", row[0])
            continue

        freq_intensities = [
            float(x) 
            for x in row[1:]
            if x.strip()  # Make sure it's not empty or whitespace
        ]

        freq_intensities = torch.tensor(freq_intensities, dtype=torch.float32)
        freq_resampled = np.linspace(1.5, 6.5, 61)  

        # 2) Interpolate your intensities onto this new axis
        intens_resampled = np.interp(freq_resampled, frequencies, freq_intensities)

        # 3) Convert back to torch if desired
        freq_resampled_torch = torch.tensor(freq_resampled,     dtype=torch.float32)
        intens_resampled_torch = torch.tensor(intens_resampled, dtype=torch.float32)

        len_freq_centers = len(intens_resampled_torch)

        """
        # Convert the Torch tensors to NumPy arrays.
        x_values = freq_centers.numpy()
        y_values = freq_intensities.numpy()

        plt.figure()
        plt.plot(x_values, y_values)
        plt.xlabel("Frequency")
        plt.ylabel("Intensity")
        plt.title("Lorentzian Spectrum")
        plt.show()
        """

        # ----------------
        # Normalization: Scale all intensities so their max is 1
        # ----------------
        if normalize:
            max_val = intens_resampled_torch.max()
            if max_val > 0:
                intens_resampled_torch = intens_resampled_torch / max_val

        freq_3d = intens_resampled_torch.unsqueeze(1).unsqueeze(2).expand(-1, 3, 1)

        mol = qm9s[idx]
        
        real_mat = mol.polar
        imag_mat = np.zeros(real_mat.shape)
        imag_mat = torch.tensor(imag_mat, dtype=torch.float32)
        complex_mat = torch.cat([real_mat, imag_mat], dim=-1)

        # Suppose complex_mat.shape == [3, 6]
        # We want to replicate it so we have a stack of 61 of those.

        complex_mat_61 = complex_mat.repeat(61, 1, 1)
        # Now complex_mat_61.shape == [61, 3, 6]


        combined = torch.cat([complex_mat_61, freq_3d], dim=2)
        data_entry = Data(
            idx = idx,
            number= mol.number,
            pos=mol.pos.to(torch.float32),    # Atomic positions
            z=torch.LongTensor(mol.z),        # Atomic numbers
            y=combined,  # Polarizability tensor (target)
        )
        dataset.append(data_entry)

ex1 = dataset[0]

print("y.shape :", ex1.y.shape)

# Train and validate per molecule
unique_mol_ids = list({data_entry.idx for data_entry in dataset})
random.shuffle(unique_mol_ids)

num_val_mols = max(1, int(0.2 * len(unique_mol_ids)))  # e.g., 20% for validation
val_mol_ids = set(unique_mol_ids[:num_val_mols])

train_datasets = [d for d in dataset if d.idx not in val_mol_ids]
val_datasets = [d for d in dataset if d.idx in val_mol_ids]

print(f"Total unique molecules: {len(unique_mol_ids)}")
#print(f"Validation molecule IDs: {sorted(val_mol_ids)}")
print(f"Training set size: {len(train_datasets)}")
print(f"Validation set size: {len(val_datasets)}")


'''Using torch_Geometric.dataloader.DataLoader Converts a dataset into a batch of 64 molecules of training data.'''

trainloader=DataLoader(train_datasets,batch_size=batch_size,shuffle=True, drop_last=True)
valloader=DataLoader(val_datasets,batch_size=batch_size,shuffle=True, drop_last=True)

wandb.init(
    # set the wandb project where this run will be logged
    project="Detanet-pol-uv-spec",
    name=f"pol-uv-spec-qm9s-beta_09_alpha_09", 
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
                    scalar_outsize=(4*61) + 61,
                    irreps_out= '122x2e',
                    summation=True,
                    norm=False,
                    out_type='complex_61_uv_tensor', # '2_tensor',
                    grad_type=None,
                    device=device)

if fine_tune:
    state_dict = torch.load("/media/maria/work_space/detanet-complex/code/trained_param/pol_uv_spec_QM9s_beta_09_alpha_09.pth")
    model.load_state_dict(state_dict=state_dict)
model.train()
model.to(device)
wandb.watch(model, log="all")


'''Finally, using the trainer, training 20 times from a 5e-4 learning rate'''
trainer=trainer_pol_uv.Trainer(model,train_loader=trainloader,val_loader=valloader,loss_function=ut.complex_uv_mse_loss,lr=lr,weight_decay=0,optimizer='AdamW')
trainer.train(num_train=epochs,targ='y')

torch.save(model.state_dict(), current_dir + f'/trained_param/pol_uv_spec_QM9s_beta_09_alpha_09.pth')



# If your trainer does NOT store them, you need to modify your Trainer class to do so.
train_losses = trainer.train_losses
val_losses = trainer.val_losses

# -----------------------------------------------------
# Create and save the loss plot for this frequency
# -----------------------------------------------------

plt.figure()
plt.plot(range(1, len(train_losses)+ 1), train_losses, label='Train Loss')
plt.plot(range(1, len(train_losses) + 1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Loss Plot')
plt.legend()
plot_path = os.path.join(current_dir, f'pol_uv_spec_QM9s_beta_09_alpha_09.png')
plt.savefig(plot_path)
plt.close()
wandb.finish()

