import csv
import json
import os
import preprocess_data
import join_datasets
import csv
import json
import os
import torch
import os.path as osp

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential

import torch_geometric.transforms as T

from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import remove_self_loops
import torch_geometric
from torch_geometric.nn import global_mean_pool


def join_polarizabilities_qm9s(
    input_str: str,
    input_csv_path: str,
    output_csv_path: str
):
    
    comp_map = {
        'xx': (0, 0), 'xy': (0, 1), 'xz': (0, 2),
        'yx': (1, 0), 'yy': (1, 1), 'yz': (1, 2),
        'zx': (2, 0), 'zy': (2, 1), 'zz': (2, 2)
    }
    

    with open(input_csv_path, "r", encoding="utf-8") as infile:
        reader = csv.reader(infile)

        tensor_line = next(reader)       # e.g. ['#tensor','ee','ee','ee','em','em','em','mm','mm','mm']
        component_line = next(reader)    # e.g. ['#component','xx','xy','xz','xx','xy','xz','xx','xy','xz']
        freq_line = next(reader)         # e.g. ['#eV','1.55','1.55','1.55','2.00','2.00','2.00',...]

        tensor_entries = tensor_line[1:]
        component_entries = component_line
        freq_entries = freq_line[1:]

        tensor_entries = [t for t in tensor_entries if t.strip()]
        component_entries = [c for c in component_entries if c.strip()]
        #freq_entries = [f for f in freq_entries if f.strip()]

        # Identify unique frequencies
        freq_values = [
            float(entry) if entry.strip() != "" else None
            for entry in freq_entries
        ]
        #freq_values = list(map(float, freq_entries))  # convert to float
        unique_freqs = []
        for fv in freq_values:
            if fv not in unique_freqs:
                unique_freqs.append(fv)
        n_freqs = len(unique_freqs)

        # n_cols = n_freqs
        n_cols = len(tensor_entries)
        blocks_info = []  # will hold dicts like {'name':'ee','size':6,'symmetry':True,'components':[...]}
        i = 0
        while i < n_cols:
            block_label = tensor_entries[i]  # e.g. 'ee'
            if block_label not in ["ee", "em", "mm"]:
                i = i + 1
                continue

            block_frequency = freq_entries[i]
            start_idx = i
            # Count how many consecutive columns share this label
            while i < n_cols and tensor_entries[i] == block_label:
                i += 1
            block_size = i - start_idx
            block_start = start_idx
            block_end = start_idx + block_size
            
            # Check if 6 => "symmetric", if 9 => "unsymmetric", else unknown
            if block_size == 6:
                blocks_info.append({
                'label': block_label,
                'frequency': block_frequency,
                'start': block_start,
                'end': block_end
                })
            elif block_size == 9:
                blocks_info.append({
                'label': block_label,
                'frequency': block_frequency,
                'start': block_start,
                'end': block_end
                })
            else:
                sym= None  

        # build csv
        frequencies = ["frequencies"]
        polarizability_type = ["polarizability_type"]

        # fill header files
        for block in blocks_info:
            frequencies.append(block["frequency"])
            polarizability_type.append(block["label"])
    
        data_rows = []
        mol_counter = 0

        for row in reader:
           
            out_row = [row[0]]
            #out_row = out_row.append(qm9_mol['z'])
            #out_row = out_row.append(qm9_mol['pos'])
            for binfo in blocks_info:
                block_start = binfo['start'] + 1 # +1 because the first entry of each row contains the mol id
                block_end = binfo['end'] + 1
                #print(f"blockstart = {block_start}; blockend = {block_end}")
                chunk = row[block_start : block_end]
                components = component_entries[block_start:block_end]

                mat = [[0j]*3 for _ in range(3)]
                
                # Fill the matrix entries that are explicitly given
                for label, val_str in zip(components, chunk):
                    # Convert string like '95.47+341.62j' to a Python complex number
                    val = val_str
                    r, c = comp_map[label]
                    mat[r][c] = val
                
                if len(components) == 6:
                    # Fill missing symmetrical entries, e.g. mat[1][0] = mat[0][1], etc.
                    # We'll just systematically check pairs we know are symmetrical.
                    # (yx, xy), (zx, xz), (zy, yz).
                    if 'xy' in components and 'yx' not in components:
                        mat[1][0] = mat[0][1]
                    if 'xz' in components and 'zx' not in components:
                        mat[2][0] = mat[0][2]
                    if 'yz' in components and 'zy' not in components:
                        mat[2][1] = mat[1][2]
                
                out_row.append(json.dumps(mat))

            data_rows.append(out_row)
            mol_counter += 1

    with open(output_csv_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(frequencies)
        writer.writerow(polarizability_type)
        writer.writerows(data_rows)

    print(f"Wrote {len(data_rows)} molecules to {output_csv_path}.")


if __name__ == "__main__":
    current_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)

    data_dir = os.path.join(grandparent_dir, 'data')
    
    path = osp.join('..', 'data', 'QM9')
    csv_input = os.path.join(data_dir, "DATA_QM9_reduced_2025_03_06.csv")
    csv_output = os.path.join(data_dir, "polarizabilities_qm9.csv")
    csv_output_ee = os.path.join(data_dir, "ee_polarizabilities_qm9s.csv")

    join_polarizabilities_qm9s(
        path,
        csv_input,
        csv_output
    )

    data = preprocess_data.load_polarizabilities(csv_output, pol_type='ee')
    preprocess_data.save_dataset_to_csv(data, csv_output_ee)
