# %%
import copy
import os
import os.path as osp
import csv
import util as ut
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

from pathlib import Path
import trainer
import json 

from detanet_model import *
import wandb

# %%
import random
random.seed(42)

# %%
batch_size = 128
epochs = 60
lr=5e-4
num_freqs=61

high_spec_cutoff = 0.1
low_fraction = 0.006

# %%

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, 'data')
csv_path = data_dir + "/ee_polarizabilities_qm9s.csv"

dataset = []
frequencies = ut.load_unique_frequencies(csv_path)

csv_path_geometries = data_dir + "/KITqm9_geometries.csv"
geometries = ut.load_geometry(csv_path_geometries)

csv_spectra = data_dir + "/DATA_QM9_reduced_2025_03_06.csv"
sprectras = ut.load_spectra(csv_spectra)

model_path = "trained_param/ee_polarizabilities_all_freq_KITqm9_smaller_than_0.000005_no_normalization.pth"

# %%
count = 0

with open(csv_path, newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    
    # Read the header to identify column indices
    header = next(csv_reader)
    frequency_idx = header.index("frequency")
    matrix_real_idx = header.index("matrix_real")
    matrix_imag_idx = header.index("matrix_imag")
    
    # Read each row
    for row in csv_reader:
        try:
            idx = int(row[0])
        except ValueError:
            print("Can't read index:", row[0])
            continue

        freq_str = row[frequency_idx]
        try:
            freq_val = float(freq_str)
        except ValueError:
            continue

        if freq_val not in frequencies:
            continue

        mol = None
        # Now you can look up any 'idx' in constant time
        if idx in geometries:
            mol = geometries[idx]
        else:
            continue        
        pos = mol.pos
        z = mol.z
        spectrum_value = ut.get_closest_spectrum_value(sprectras, idx, freq_val)

        # Parse JSON for real matrix
        matrix_real_str = row[matrix_real_idx]
        matrix_imag_str = row[matrix_imag_idx]
        try:
            real_3x3 = json.loads(matrix_real_str)  # expected shape [3,3]
        except json.JSONDecodeError:
            print("Warning: Could not parse real part of matrix for idx:", idx)
            continue

        try:
            imag_3x3 = json.loads(matrix_imag_str)  # expected shape [3,3]
        except json.JSONDecodeError:
            print("Warning: Could not parse imaginary part of matrix for idx:", idx)
            continue

        real_mat = torch.tensor(real_3x3, dtype=torch.float32)
        imag_mat = torch.tensor(imag_3x3, dtype=torch.float32)
        
        y = torch.cat([real_mat, imag_mat], dim=-1)  # shape [12]
            
        data_entry = Data(
            idx = mol.idx,
            pos=pos.to(torch.float32),    # Atomic positions
            z=torch.LongTensor(z),        # Atomic numbers
            freq=torch.tensor(float(freq_val), dtype=torch.float32),
            spec=torch.tensor(float(spectrum_value), dtype=torch.float32),
            y=y,  # Polarizability tensor (target)
        )
        if spectrum_value < 0.000005:#> high_spec_cutoff:
            dataset.append(data_entry)
            count += 1
        #else:
         #   # Randomly sample ~0.2% of the "low-spec" data
          #  if random.random() < low_fraction:
           #     dataset.append(data_entry)
                

# %%

print(f"Collected {count} high-spec (>0.1) entries.")
print(f"Total dataset length: {len(dataset)}")

ex1 = dataset[0]
ex2 = dataset[5]

print("dataset[0] :", ex1.idx, ex1.freq, ex1.spec)
print("dataset[5] :", ex2.idx, ex2.freq, ex2.spec)

# %%
spec_values = [item.spec.item() for item in dataset]
spec_mean = np.mean(spec_values)
spec_std = np.std(spec_values)

print("Spec mean, std =", spec_mean, spec_std)
for item in dataset:
    old_val = item.spec.item()
    norm_val = (old_val - spec_mean) / (spec_std + 1e-8)  # avoid div by zero
    item.spec = torch.tensor(norm_val, dtype=torch.float32)

# %%
"""
import numpy as np

y_vals = []

for item in dataset:
    y = item.y.reshape(-1).tolist()   
    y_vals.extend(y)
# compute mean, std
y_mean, y_std = np.mean(y_vals), np.std(y_vals)

print("y mean, std =", y_mean, y_std)

# Now transform each data entry
for item in dataset:
    y = item.y  # [3,6]
    # 4) do standard z-score
    y_norm = (y - y_mean)/(y_std + 1e-8)
    item.y = y_norm"
"""

# %%
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
                    max_atomic_number=9,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize= 4, # 2,#4, 
                    irreps_out= '2x2e', #'2e',# '2e+2e',
                    summation=True,
                    norm=False,
                    out_type='complex_2_tensor', # '2_tensor',
                    grad_type=None,
                    device=device)

# %%
state_dict = torch.load(model_path)
model.load_state_dict(state_dict=state_dict)
model.to(device)

# %%
sample = dataset[1]


# %%
import torch_cluster
import ase
from ase.io import read
from ase.visualize import view
from ase.build import molecule
#from code.util.visualize_polarizability import smiles_to_atoms, visualize_polarizability, compare_polarizabilities_eigen

result = model(pos=sample.pos.to(device), z=sample.z.to(device), spec=sample.spec.to(device))
print(result)



# %%
sample.y

# %%
from ase.build import molecule

qm9s = torch.load("../data/qm9s.pt")

# %%
qm9s_dict = {entry.number: entry for entry in qm9s}

# %%
print(qm9s_dict[sample.idx].number)
print(qm9s_dict[sample.idx].z)
print(qm9s_dict[sample.idx].smile)
print(sample.idx)
print(sample.z)



# %%
from util.visualize_polarizability import smiles_to_atoms, visualize_polarizability, compare_polarizabilities_eigen
atoms = smiles_to_atoms(qm9s_dict[sample.idx].smile)
view(atoms, viewer='x3d')

# %%
real_part_true = sample.y[:,:3]
print(real_part_true)

visualize_polarizability(atoms, real_part_true)

real_part_predicted = result[:,:3]
print(real_part_predicted)

# %%
import plotly.graph_objects as go

def compare_polarizabilities(atoms, true_tensor, predicted_tensor):
    """
    Visualize a molecule with true and predicted polarizability tensors using Plotly.
    
    Parameters:
        atoms: ASE Atoms object
        true_tensor: (3, 3) numpy array representing the true polarizability tensor (Bohr^3)
        predicted_tensor: (3, 3) numpy array representing the predicted polarizability tensor (Bohr^3)
    """
    def create_polarizability_traces(tensor, center, color, name, scale=0.05):

        tensor = tensor.reshape(3,3)

        # Eigen decomposition of the polarizability tensor
        eigenvalues, eigenvectors = np.linalg.eig(tensor)
        
        print("Raw Eigenvalues (True Tensor):", eigenvalues)
        print("Real Eigenvalues (True Tensor):", np.real(eigenvalues))
        # Ensure eigenvalues and eigenvectors are real
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)


        # Calculate arrows for the principal axes
        arrows = [scale * eigenvalue * eigenvector for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors.T)]

        # Create arrow traces
        arrow_traces = []
        for arrow in arrows:
            arrow_trace = go.Scatter3d(
                x=[center[0], center[0] + arrow[0]],
                y=[center[1], center[1] + arrow[1]],
                z=[center[2], center[2] + arrow[2]],
                mode='lines+markers',
                line=dict(color=color, width=4),
                marker=dict(size=4, color=color),
                name=f"{name} Arrow"
            )
            arrow_traces.append(arrow_trace)

        # Generate ellipsoid points
        eigenvalues = np.abs(eigenvalues) * 0.05  # Adjust scaling factor
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 25)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        # Scale and rotate the ellipsoid
        ellipsoid = np.dot(np.column_stack([x.flatten(), y.flatten(), z.flatten()]), np.diag(eigenvalues))
        ellipsoid = np.dot(ellipsoid, eigenvectors.T)
        x_ellipsoid = ellipsoid[:, 0].reshape(x.shape)
        y_ellipsoid = ellipsoid[:, 1].reshape(y.shape)
        z_ellipsoid = ellipsoid[:, 2].reshape(z.shape)

        # Create ellipsoid trace
        ellipsoid_trace = go.Surface(
            x=x_ellipsoid + center[0],
            y=y_ellipsoid + center[1],
            z=z_ellipsoid + center[2],
            colorscale=[[0, color], [1, color]],
            opacity=0.2,
            showscale=False,
            name=f"{name} Ellipsoid"
        )
        return arrow_traces, ellipsoid_trace

    # Calculate center of the molecule
    center = atoms.get_center_of_mass()

    # Create polarizability traces for true tensor
    true_arrows, true_ellipsoid = create_polarizability_traces(true_tensor, center, 'blue', 'True')

    # Create polarizability traces for predicted tensor
    pred_arrows, pred_ellipsoid = create_polarizability_traces(predicted_tensor, center, 'red', 'Predicted')

    # Extract atomic positions and symbols
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    # Define colors for different atom types
    atom_colors = {
        'H': 'grey',
        'C': 'black',
        'O': 'pink',
        'N': 'purple',
        'S': 'yellow',
        'Cl': 'green',
        'F': 'cyan'
    }
    colors = [atom_colors.get(symbol, 'gray') for symbol in symbols]

    # Create scatter plot for the atoms with different colors
    atom_trace = go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode='markers+text',
        marker=dict(size=6, color=colors),
        text=symbols,
        textposition="top center",
        name="Atoms"
    )

    # Combine all traces
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        title="True vs Predicted Polarizability Comparison"
    )

    fig = go.Figure(data=[atom_trace, true_ellipsoid, pred_ellipsoid] + true_arrows + pred_arrows, layout=layout)

    # Display the plot
    fig.show()

# %%
compare_polarizabilities(
                atoms,
                np.array(real_part_true.cpu().detach()),
                np.array(real_part_predicted.cpu().detach())
            )

# %%



