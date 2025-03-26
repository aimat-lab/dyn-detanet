from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data
import json
import csv
import numpy as np
import pandas as pd
import csv
from functools import partial
import math

# Function to process SMILES into atomic numbers and 3D positionn
def smiles_to_graph(smiles):
    """
    Convert a SMILES string into atomic numbers (z) and 3D coordinates (pos) using RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Invalid SMILES string

    mol = Chem.AddHs(mol)  # Add hydrogens
    result = AllChem.EmbedMolecule(mol)
    if result != 0:
        # Embedding failed; try with random coordinates
        result = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if result != 0:
            print(f"Computation for {smiles} failed")
            return None  # Embedding failed again

    AllChem.UFFOptimizeMolecule(mol)  # Optimize geometry
    # Extract atomic numbers
    z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
    
    # Extract 3D positions
    conf = mol.GetConformer()
    pos = torch.tensor([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=torch.float)
    
    return z, pos



import torch.nn.functional as F

def complex_mse_loss(pred, target, alpha_param=0.5):

    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} vs {target.shape}")

    #print("pred_shape", pred.shape)
    #print("target", target)

    #print("pred.shape ", pred.shape)

    real_pred = pred[:, :3]   # shape [BatchSize, 3]
    imag_pred = pred[:, 3:]   # shape [BatchSize, 3]
    real_targ = target[:, :3]
    imag_targ = target[:, 3:]

    #print("real_pred", real_pred)
    #print("imag_pred", imag_pred)
    #print("real_targ", real_targ)
    #print("imag_targ", imag_targ)

    loss_real = F.mse_loss(real_pred, real_targ)
    loss_imag = F.mse_loss(imag_pred, imag_targ)

    #print("loss_real", loss_real)
    #print("loss_imag", loss_imag)
  
    loss = (1 - alpha_param) * loss_real + alpha_param * loss_imag

    return loss


def fun_complex_mse_loss(pred, target):
    return complex_mse_loss(pred, target) 




def compute_polarizability_stats(csv_path, qm9_dict, matrix_real_idx=None, matrix_imag_idx=None):
    """
    Reads all entries from csv_path, collects real & imaginary polarizabilities
    into arrays, and computes mean/std for each part.

    Args:
        csv_path (str): Path to the polarizability CSV file.
        qm9_dict (dict): Dictionary keyed by molecule index -> PyG Data objects.
        matrix_real_idx (int): Column index for the real part. If None, it's inferred from CSV header.
        matrix_imag_idx (int): Column index for the imag part. If None, it's inferred from CSV header.

    Returns:
        real_mean (np.ndarray): shape [9], mean of real part
        real_std (np.ndarray): shape [9], std of real part
        imag_mean (np.ndarray): shape [9], mean of imag part
        imag_std (np.ndarray): shape [9], std of imag part
    """
    all_real_parts = []
    all_imag_parts = []

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        header = next(csv_reader)

        # If not provided, find the columns by name
        if matrix_real_idx is None:
            matrix_real_idx = header.index("matrix_real")
        if matrix_imag_idx is None:
            matrix_imag_idx = header.index("matrix_imag")

        for row in csv_reader:
            # Safely parse molecule index
            try:
                idx = int(row[0])
            except ValueError:
                continue

            # Skip if not in qm9_dict
            if idx not in qm9_dict:
                continue

            # Parse real & imaginary strings
            real_str = row[matrix_real_idx]
            imag_str = row[matrix_imag_idx]
            try:
                real_3x3 = json.loads(real_str)  # shape [3,3]
                imag_3x3 = json.loads(imag_str)  # shape [3,3]
            except json.JSONDecodeError:
                continue

            # Flatten to shape [9]
            real_mat = torch.tensor(real_3x3, dtype=torch.float32).view(-1)
            imag_mat = torch.tensor(imag_3x3, dtype=torch.float32).view(-1)

            all_real_parts.append(real_mat.numpy())
            all_imag_parts.append(imag_mat.numpy())

    if not all_real_parts:
        # No valid data found; return zeros to avoid errors
        # or raise an Exception, depending on your preference
        return np.zeros(9), np.ones(9), np.zeros(9), np.ones(9)

    # Convert to numpy
    all_real_parts = np.array(all_real_parts)  # shape [N, 9]
    all_imag_parts = np.array(all_imag_parts)  # shape [N, 9]

    # Compute mean, std (add a small eps to std to avoid division by 0)
    eps = 1e-12
    real_mean = all_real_parts.mean(axis=0)
    real_std = all_real_parts.std(axis=0) + eps
    imag_mean = all_imag_parts.mean(axis=0)
    imag_std = all_imag_parts.std(axis=0) + eps

    return real_mean, real_std, imag_mean, imag_std


def normalize_polarizability(real_3x3, imag_3x3, real_mean, real_std, imag_mean, imag_std):
    """
    Normalizes the real and imaginary parts of the 3x3 polarizability
    using separate mean/std for each part, then returns a tensor of
    shape [3, 6]:
      - The first 3 columns are the real part (normalized),
      - The last 3 columns are the imaginary part (normalized).
    """

    real_3x3_np = np.array(real_3x3, dtype=np.float32)  # shape [3,3]
    imag_3x3_np = np.array(imag_3x3, dtype=np.float32)  # shape [3,3]

    # Flatten => shape [9]
    real_mat_flat = real_3x3_np.flatten()
    imag_mat_flat = imag_3x3_np.flatten()

    # Standard scaling: (value - mean) / std
    real_norm_flat = (real_mat_flat - real_mean) / real_std  # shape [9]
    imag_norm_flat = (imag_mat_flat - imag_mean) / imag_std  # shape [9]

    # Convert back to torch and reshape each one to [3,3]
    real_norm_3x3 = torch.from_numpy(real_norm_flat).float().view(3, 3)  # shape [3,3]
    imag_norm_3x3 = torch.from_numpy(imag_norm_flat).float().view(3, 3)  # shape [3,3]

    # Concatenate them along the columns => shape [3, 6]
    y_norm_3x6 = torch.cat([real_norm_3x3, imag_norm_3x3], dim=-1)  # shape [3,6]

    return y_norm_3x6



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


def load_unique_frequencies(csv_path):
    frequencies = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        header = next(csv_reader)
        freq_idx = header.index("frequency")

        for row in csv_reader:
            if not row:
                continue
            try:
                f_val = float(row[freq_idx])
                frequencies.append(f_val)
            except ValueError:
                # skip invalid freq
                pass
    return list(set(frequencies))


def load_spectra(csv_path, fun_type="l"):
    dataset_dict = {}
    with open(csv_path, "r", encoding="utf-8") as infile:
        reader = csv.reader(infile)

        type_line = next(reader)       
        empty_line = next(reader)   
        freq_line = next(reader)         

        type_entries = type_line[1:]
        freq_entries = freq_line[1:]

        freq_values = []
        for x in freq_entries:
            try:
                freq_values.append(float(x))
            except ValueError:
                # If conversion fails, store None
                freq_values.append(None)

    
        mask = []
        # Build a mask that selects only columns of interest
        if fun_type == "l":
            # Indices of columns whose type is "spec. L 0.05eV"
            mask = [i for i, t in enumerate(type_entries) if t == "spec. L 0.05eV"]
        elif fun_type == "g":
            # Indices of columns whose type is "spec. G 0.5eV"
            mask = [i for i, t in enumerate(type_entries) if t == "spec. G 0.5eV"]
        else:
            raise ValueError("Please use fun_type='l' (Lorentzian) or fun_type='g' (Gaussian).")

        # Get just the frequencies that correspond to the masked columns
        selected_freqs = [freq_values[i] for i in mask]

        for row in reader:
            if not row:
                continue  # skip empty lines

            idx = int(row[0])
            # Remove the first entry if it's just a label or ID (depends on your CSV format)
            row_vals = row[1:]  # skip the leftmost column if it’s not numeric
            
            row_vals = [row_vals[i] for i in mask]
            numeric_vals = [float(v) for v in row_vals]
            dataset_dict[idx] = {'frequencies': selected_freqs, 'values': numeric_vals}
        
    return dataset_dict

        
def get_closest_spectrum_value(dataset_dict, molecule_id, target_freq):
    """
    Finds the closest frequency to `target_freq` in the spectrum of `molecule_id`
    and returns the corresponding value.

    Parameters:
    - dataset_dict: The dataset dictionary created by `load_spectra`.
    - molecule_id: The string identifier for the molecule (as in the CSV).
    - target_freq: The frequency you want to look up.

    Returns:
    - (closest_freq, value_at_closest_freq)
    """
    if molecule_id not in dataset_dict:
        raise KeyError(f"Molecule ID '{molecule_id}' not found in dataset.")

    freqs = dataset_dict[molecule_id]['frequencies']
    values = dataset_dict[molecule_id]['values']

    # Find index of closest frequency
    closest_index = min(range(len(freqs)), key=lambda i: abs(freqs[i] - target_freq))

    return values[closest_index]




def lorentzian_func(x, center, amplitude, gamma=0.05):
    """
    Basic Lorentzian function:
      amplitude / [1 + ((x - center) / gamma)^2]
    """
    return amplitude / (1.0 + ((x - center) / gamma)**2)

def gaussian_func(x, center, amplitude, sigma=0.5):
    """
    Basic Gaussian function:
      amplitude * exp( -((x - center)^2 / (2 * sigma^2)) )
    """
    exponent = -((x - center)**2) / (2.0 * sigma**2)
    return amplitude * math.exp(exponent)

def load_spectra_with_profiles(csv_path, fun_type="l"):
    """
    Reads the CSV and constructs a list of line-shape functions (Lorentzian or
    Gaussian) for each row's (frequency, intensity) pair. The dictionary entry
    for each molecule ID will contain:
      {
         'line_shapes': [callable, callable, ...], 
         'freq_centers': [f1, f2, ...],  # The center frequencies
      }

    :param csv_path: path to your CSV file
    :param fun_type: either 'l' for Lorentzian or 'g' for Gaussian in the CSV
    :return: dict of molecule_id -> { 'line_shapes': [...], 'freq_centers': [...] }
    """
    dataset_dict = {}

    with open(csv_path, "r", encoding="utf-8") as infile:
        reader = csv.reader(infile)

        # The first line: e.g. "ID, spec. L 0.05eV, spec. L 0.05eV, ..."
        type_line = next(reader)
        empty_line = next(reader) 
        # The third line: e.g. "ID, 1.5, 1.55, 1.6, 1.65, ..."
        freq_line = next(reader)

        # Actual text headers after "ID"
        type_entries = type_line[1:]
        # Frequencies from the line after "ID"
        freq_entries = freq_line[1:]

        # Convert the frequency headers into floats (some might be empty)
        freq_values = []
        for x in freq_entries:
            try:
                freq_values.append(float(x))
            except ValueError:
                freq_values.append(None)

        # Build a mask that selects only columns of interest (Lorentzian or Gaussian)
        if fun_type == "l":
            mask = [i for i, t in enumerate(type_entries) if t == "spec. L 0.05eV"]
            width_param = 0.05   # gamma for Lorentzian
            is_lorentzian = True
        elif fun_type == "g":
            mask = [i for i, t in enumerate(type_entries) if t == "spec. G 0.5eV"]
            width_param = 0.5    # sigma for Gaussian
            is_lorentzian = False
        else:
            raise ValueError("Please use fun_type='l' (Lorentzian) or fun_type='g' (Gaussian).")

        # Keep only the masked columns (our relevant frequencies)
        selected_freqs = [freq_values[i] for i in mask]

        # Now read the rest of the file for data rows
        for row in reader:
            if not row:
                continue  # skip empty lines

            idx_str = row[0].strip()
            if not idx_str.isdigit():
                # In case there's an unexpected string instead of an integer ID
                continue

            idx = int(idx_str)

            # Next columns are the intensities
            row_vals = row[1:]
            # Filter by the same mask
            row_vals = [row_vals[i] for i in mask]

            # Convert them to floats (the intensities)
            numeric_vals = [float(v) for v in row_vals]

            # Create line-shape callables for each frequency/intensity pair
            # partial(...) => returns a function f(x)
            if is_lorentzian:
                line_shapes = [
                    partial(lorentzian_func, center=f, amplitude=a, gamma=width_param)
                    for f, a in zip(selected_freqs, numeric_vals)
                ]
            else:
                line_shapes = [
                    partial(gaussian_func, center=f, amplitude=a, sigma=width_param)
                    for f, a in zip(selected_freqs, numeric_vals)
                ]

            dataset_dict[idx] = {line_shapes}

    return dataset_dict