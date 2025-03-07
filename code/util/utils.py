from rdkit import Chem
from rdkit.Chem import AllChem
import torch

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