from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import json
import csv
import numpy as np

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

