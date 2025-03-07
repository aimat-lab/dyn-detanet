#https://colab.research.google.com/drive/1duR1Y-roE_CSL3hrGxINMZ4XIepHvoHt?usp=sharing#scrollTo=4MLlkDWf7jM5
import torch
import ase
from ase.io import read
from ase.visualize import view
from e3nn_invstutorial.orthonormal_radial_basis import (
    FixedCosineRadialModel,
    CosineFunctions,
    FadeAtCutoff,
    OrthonormalRadialFunctions
    )
from e3nn_invstutorial.radial_spherical_tensor import *
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import torch.autograd.functional

torch.set_default_dtype(torch.float32)

#Data(edge_index=[2, 418], pos=[23, 3], number=57550, smile='CC(C)(C)CNCC#N', z=[23], quadrupole=[1, 3, 3], octapole=[1, 3, 3, 3], npacharge=[23], dipole=[1, 3], polar=[1, 3, 3], hyperpolar=[1, 3, 3, 3], energy=[1, 1], Hij=[418, 3, 3], Hi=[23, 3, 3], dedipole=[23, 3, 3], depolar=[23, 3, 6], tran_dipole=[1, 10, 3], tran_energy=[1, 10])

from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.visualize import view


def smiles_to_atoms(smiles):
    """Convert a SMILES string to an ASE Atoms object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    
    # Add hydrogens to the molecule
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    
    # Extract atomic positions and symbols
    positions = mol.GetConformers()[0].GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    
    # Create ASE Atoms object
    return Atoms(symbols=symbols, positions=positions)


def visualize_polarizability(atoms, polarizability_tensor):
    """
    Visualize a molecule with polarizability represented as arrows using Plotly.
    
    Parameters:
        atoms: ASE Atoms object
        polarizability_tensor: (3, 3) numpy array representing the polarizability tensor (Bohr^3)
    """
    # Extract the first tensor if it's wrapped in a batch-like structure
    if polarizability_tensor.ndim == 3:
        polarizability_tensor = polarizability_tensor[0]
    
    # Define scale factor for arrows
    scale = 0.05  # Adjust for visibility
    
    # Calculate center of the molecule for visualizing the tensor
    center = atoms.get_center_of_mass()

    # Eigen decomposition of the polarizability tensor
    eigenvalues, eigenvectors = np.linalg.eig(polarizability_tensor)


    print("Raw Eigenvalues (True Tensor):", eigenvalues)
    print("Real Eigenvalues (True Tensor):", np.real(eigenvalues))
    # Ensure eigenvalues and eigenvectors are real
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # Calculate arrows for the principal axes
    arrows = [scale * eigenvalue * eigenvector for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors.T)]

    # Extract atomic positions and symbols
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    # Define colors for different atom types
    atom_colors = {
        'H': 'grey',
        'C': 'black',
        'O': 'pink',
        'N': 'blue',
        'S': 'yellow',
        'Cl': 'green',
        'F': 'cyan'
        # Add more elements as needed
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
    
    # Create arrow traces for the polarizability tensor
    arrow_traces = []
    for arrow in arrows:
        arrow_trace = go.Scatter3d(
            x=[center[0], center[0] + arrow[0]],
            y=[center[1], center[1] + arrow[1]],
            z=[center[2], center[2] + arrow[2]],
            mode='lines+markers',
            line=dict(color='red', width=4),  # Arrow shaft
            marker=dict(size=4, color='red'),  # Optional marker at arrowhead
            name="Polarizability Arrow"
        )
        arrow_traces.append(arrow_trace)

    ### ELLIPSOID
    # Scale eigenvalues for visualization
    eigenvalues = np.abs(eigenvalues) * 0.05  # Adjust scaling factor

    # Generate ellipsoid points
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Scale ellipsoid by eigenvalues and rotate by eigenvectors
    ellipsoid = np.dot(np.column_stack([x.flatten(), y.flatten(), z.flatten()]), np.diag(eigenvalues))
    ellipsoid = np.dot(ellipsoid, eigenvectors.T)
    x_ellipsoid = ellipsoid[:, 0].reshape(x.shape)
    y_ellipsoid = ellipsoid[:, 1].reshape(y.shape)
    z_ellipsoid = ellipsoid[:, 2].reshape(z.shape)

    # Extract atomic positions and symbols
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    # Define colors for different atom types
    atom_colors = {
        'H': 'grey',
        'C': 'black',
        'O': 'purple',
        'N': 'blue',
        'S': 'yellow',
        'Cl': 'green',
        'F': 'cyan'
        # Add more elements as needed
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

    # Create mesh for the ellipsoid
    ellipsoid_trace = go.Surface(
        x=x_ellipsoid + atoms.get_center_of_mass()[0],
        y=y_ellipsoid + atoms.get_center_of_mass()[1],
        z=z_ellipsoid + atoms.get_center_of_mass()[2],
        colorscale="Reds",
        opacity=0.5,
        showscale=False,
        name="Polarizability Ellipsoid"
    )

    # Combine the atom and arrow traces
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        ),
        title="Molecule with Polarizability Arrows"
    )
    fig = go.Figure(data=[atom_trace, ellipsoid_trace] + arrow_traces, layout=layout)

    # Display the plot
    fig.show()



def compare_polarizabilities_eigen(atoms, true_eigenvalues, true_eigenvectors, pred_eigenvalues, pred_eigenvectors):
    """
    Visualize a molecule with true and predicted polarizability tensors using eigenvalues and eigenvectors.

    Parameters:
        atoms: ASE Atoms object
        true_eigenvalues: (3,) numpy array representing the eigenvalues of the true polarizability tensor.
        true_eigenvectors: (3, 3) numpy array representing the eigenvectors of the true polarizability tensor.
        pred_eigenvalues: (3,) numpy array representing the eigenvalues of the predicted polarizability tensor.
        pred_eigenvectors: (3, 3) numpy array representing the eigenvectors of the predicted polarizability tensor.
    """

    def create_polarizability_traces(eigenvalues, eigenvectors, center, color, name, scale=0.05):
        # Ensure eigenvalues and eigenvectors are real
        eigenvalues = np.array(eigenvalues)
        eigenvectors = np.array(eigenvectors).reshape(3, 3)

        center = np.array(center)
        if center.shape != (3,):
            raise ValueError(f"Center must be a (3,) array, but got shape {center.shape}")


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
        eigenvalues_scaled = np.abs(eigenvalues) * 0.05  # Adjust scaling factor
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 25)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))

        # Scale and rotate the ellipsoid
        ellipsoid = np.dot(np.column_stack([x.flatten(), y.flatten(), z.flatten()]), np.diag(eigenvalues_scaled))
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
            opacity=0.5,
            showscale=False,
            name=f"{name} Ellipsoid"
        )
        return arrow_traces, ellipsoid_trace
    
    print("In method")

    # Calculate center of the molecule
    center = atoms.get_center_of_mass()

    print("center", center)

    # Create polarizability traces for true tensor
    true_arrows, true_ellipsoid = create_polarizability_traces(true_eigenvalues, true_eigenvectors, center, 'blue', 'True')

    # Create polarizability traces for predicted tensor
    pred_arrows, pred_ellipsoid = create_polarizability_traces(pred_eigenvalues, pred_eigenvectors, center, 'red', 'Predicted')

    # Extract atomic positions and symbols
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    # Define colors for different atom types
    atom_colors = {
        'H': 'grey',
        'C': 'black',
        'O': 'pink',
        'N': 'blue',
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
