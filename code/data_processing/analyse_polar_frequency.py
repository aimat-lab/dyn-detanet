import csv
import json
import random
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.pyplot as plt
import random

def load_polar_data(csv_path):
    """
    Returns a dict:
      { smiles_str : [ (freq, real_3x3, imag_3x3), ..., ... ] }
    where real_3x3 and imag_3x3 are 3x3 lists (or None if you only parse real).
    """
    data_dict = defaultdict(list)
    print("Load data")
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)
        
        # Indices
        smiles_idx = header.index("smiles")
        freq_idx = header.index("frequency")
        matrix_real_idx = header.index("matrix_real")
        matrix_imag_idx = header.index("matrix_imag")

        for row in reader:
            smiles_str = row[smiles_idx]

            # parse frequency
            try:
                freq_val = float(row[freq_idx])
            except ValueError:
                continue

            # parse real polarizability matrix
            matrix_real_str = row[matrix_real_idx]
            try:
                real_3x3 = json.loads(matrix_real_str)  # shape ~ [3,3]
            except json.JSONDecodeError:
                continue

            # parse imaginary polarizability matrix (if you want it)
            imag_3x3 = None
            if matrix_imag_idx < len(row):
                matrix_imag_str = row[matrix_imag_idx]
                try:
                    imag_3x3 = json.loads(matrix_imag_str)
                except json.JSONDecodeError:
                    imag_3x3 = None  # skip if malformed

            # store it
            data_dict[smiles_str].append((freq_val, real_3x3, imag_3x3))
    print("Data loaded")

    return data_dict

def plot_trace_vs_frequency(data_dict, value='real', max_mols=2):
    """
    Plots trace(real_3x3) vs. frequency for up to `max_mols` molecules from data_dict.
    If you want to do them all, set max_mols=None, but it may get busy.
    """
    # If max_mols is not None, pick random molecules
    smiles_list = list(data_dict.keys())
    if max_mols is not None:
        # pick up to max_mols random ones
        if len(smiles_list) > max_mols:
            smiles_list = random.sample(smiles_list, k=max_mols)

    # Create a separate plot or a single plot with multiple lines
    plt.figure(figsize=(7,5))

    for smi in smiles_list:
        freq_vals = []
        trace_vals = []

        for (freq, real_3x3, imag_3x3) in data_dict[smi]:
            
            if value == 'real':
                mat = real_3x3
                if real_3x3 is None:
                    continue
            else:
                value = 'imag'
                mat = imag_3x3
                if real_3x3 is None:
                    continue

            diag_sum = mat[0][0] + mat[1][1] + mat[2][2]

            
            freq_vals.append(freq)
            trace_vals.append(diag_sum)

        # sort by freq so the line plot doesn't jump around
        combined = sorted(zip(freq_vals, trace_vals), key=lambda x: x[0])
        sorted_freqs = [c[0] for c in combined]
        sorted_trace_re = [c[1] for c in combined]

        plt.plot(sorted_freqs, sorted_trace_re, marker='o', label=smi)

    plt.title(f"Trace of Polarizability vs Frequency {value}")
    plt.xlabel("Frequency")
    plt.ylabel(f"Trace({value} Polarizability)")
    if max_mols is not None and max_mols <= 10:
        plt.legend()
    plt.savefig(f"Trace_Polarizability_Frequency{value}.png")    

import matplotlib.pyplot as plt
import random

def plot_trace_vs_frequency_real_vs_imag(data_dict, max_mols=2):
    """
    Plots the real and imaginary traces of the polarizability vs. frequency for up to `max_mols` molecules 
    from data_dict. If you want to do them all, set max_mols=None.
    """
    smiles_list = list(data_dict.keys())
    if max_mols is not None:
        # Randomly pick up to max_mols molecules
        if len(smiles_list) > max_mols:
            smiles_list = random.sample(smiles_list, k=max_mols)

    plt.figure(figsize=(7, 5))

    # Get a default color cycle to ensure consistent "similar" colors for each molecule
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, smi in enumerate(smiles_list):
        freq_vals = []
        trace_vals_re = []
        trace_vals_im = []

        # Collect data
        for (freq, real_3x3, imag_3x3) in data_dict[smi]:
            if real_3x3 is None or imag_3x3 is None:
                continue

            diag_sum_re = real_3x3[0][0] + real_3x3[1][1] + real_3x3[2][2]
            diag_sum_im = imag_3x3[0][0] + imag_3x3[1][1] + imag_3x3[2][2]

            freq_vals.append(freq)
            trace_vals_re.append(diag_sum_re)
            trace_vals_im.append(diag_sum_im)

        # Sort by frequency so the plot lines do not jump around
        combined = sorted(zip(freq_vals, trace_vals_re, trace_vals_im), key=lambda x: x[0])
        sorted_freqs = [c[0] for c in combined]
        sorted_trace_re = [c[1] for c in combined]
        sorted_trace_im = [c[2] for c in combined]

        # Pick a base color for this molecule
        base_color = color_cycle[i % len(color_cycle)]

        # Plot real part
        plt.plot(sorted_freqs, sorted_trace_re,
                 color=base_color,
                 marker='o',
                 label=f"{smi} (Real)")

        # Plot imaginary part in a similar color but visually distinct (lighter/dashed/alpha)
        plt.plot(sorted_freqs, sorted_trace_im,
                 color=base_color,
                 linestyle='--',
                 marker='x',
                 alpha=0.6,
                 label=f"{smi} (Imag)")

    plt.title("Trace of Polarizability vs. Frequency")
    plt.xlabel("Frequency")
    plt.ylabel("Trace(Polarizability)")
    if max_mols is None or (max_mols is not None and max_mols <= 10):
        plt.legend()
    plt.savefig(f"Trace_Polarizability_Frequency.png")
    #plt.show()


def plot_matrix_elements_vs_freq(data_dict, max_mols=2):
    """
    Plots each real_3x3 matrix element vs. frequency on a 3x3 grid of subplots.
    Will choose up to `max_mols` random molecules. If there are fewer than that,
    it just plots them all.
    """
    # 1) Select up to max_mols random molecules
    smiles_list = list(data_dict.keys())
    if len(smiles_list) > max_mols:
        smiles_list = random.sample(smiles_list, k=max_mols)

    # 2) Prepare a 3x3 grid of subplots
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    fig.suptitle("Matrix Elements vs Frequency")

    # For each chosen molecule, gather freq vs each element
    for smi in smiles_list:
        # Gather data
        freq_vals = []
        mats = []
        for (freq, mat3x3, _) in data_dict[smi]:
            freq_vals.append(freq)
            mats.append(mat3x3)

        # Sort by frequency so lines plot in ascending order
        combined = sorted(zip(freq_vals, mats), key=lambda x: x[0])
        freq_sorted = [c[0] for c in combined]
        mats_sorted = [c[1] for c in combined]

        # 3) For each subplot (i, j), plot freq vs mat[i][j]
        for i in range(3):
            for j in range(3):
                ax = axes[i, j]
                # Build a list of that element across all data points
                elem_values = [mat_ij[i][j] for mat_ij in mats_sorted]
                # Plot
                ax.plot(freq_sorted, elem_values, marker='o', label=smi)
                ax.set_xlabel("Frequency")
                ax.set_ylabel(f"mat[{i}][{j}]")

    # If few molecules, we can add a legend
    if max_mols <= 5:
        # Put legend on any subplot, e.g. top-left:
        axes[0,0].legend()

    plt.tight_layout()
    plt.savefig("Matrix_Elements_vs_Frequency")
    plt.show()



def plot_matrix_elements_vs_freq_real_vs_imag(data_dict, max_mols=2):
    """
    Plots each real_3x3 and imag_3x3 matrix element vs. frequency on a 3x3 grid of subplots,
    for up to `max_mols` molecules chosen from data_dict.
    """
    # 1) Select up to max_mols random molecules
    smiles_list = list(data_dict.keys())
    if max_mols is not None and len(smiles_list) > max_mols:
        smiles_list = random.sample(smiles_list, k=max_mols)

    # 2) Prepare a 3x3 grid of subplots
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
    fig.suptitle("Matrix Elements (Real & Imag) vs Frequency")

    # A color cycle to keep real/imag pairs visually similar for each molecule
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # For each chosen molecule, gather freq vs each real/imag matrix
    for m_idx, smi in enumerate(smiles_list):
        freq_vals = []
        mats_real = []
        mats_imag = []

        for (freq, mat3x3_real, mat3x3_imag) in data_dict[smi]:
            freq_vals.append(freq)
            mats_real.append(mat3x3_real)
            mats_imag.append(mat3x3_imag)

        # Sort by frequency so lines plot in ascending order
        combined = sorted(zip(freq_vals, mats_real, mats_imag), key=lambda x: x[0])
        freq_sorted = [c[0] for c in combined]
        real_sorted = [c[1] for c in combined]
        imag_sorted = [c[2] for c in combined]

        # Base color for the real part; dashed/alpha version for the imaginary
        base_color = color_cycle[m_idx % len(color_cycle)]

        # 3) For each subplot (i, j), plot freq vs mat[i][j]
        for i in range(3):
            for j in range(3):
                ax = axes[i, j]

                elem_values_re = [mat[i][j] for mat in real_sorted]
                elem_values_im = [mat[i][j] for mat in imag_sorted]

                # Plot real part
                ax.plot(freq_sorted, elem_values_re,
                        color=base_color,
                        marker='o',
                        label=f"{smi} (Re)" if (i == 0 and j == 0) else None)

                # Plot imaginary part
                ax.plot(freq_sorted, elem_values_im,
                        color=base_color,
                        linestyle='--',
                        marker='x',
                        alpha=0.6,
                        label=f"{smi} (Im)" if (i == 0 and j == 0) else None)

                ax.set_xlabel("Frequency")
                ax.set_ylabel(f"mat[{i}][{j}]")

    # If few molecules, show legend in the top-left subplot
    if max_mols is None or (max_mols is not None and max_mols <= 5):
        axes[0, 0].legend()

    plt.tight_layout()
    plt.savefig("Matrix_Elements_vs_Frequency_real_vs_imag.png")
    plt.show()



if __name__ == "__main__":
    current_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, 'data')

    csv_path = data_dir + "/ee_polarizabilities.csv"
    data_dict = load_polar_data(csv_path)

    # Plot for just 2 random molecules
    plot_matrix_elements_vs_freq_real_vs_imag(data_dict, max_mols=1)

    plot_trace_vs_frequency_real_vs_imag(data_dict, max_mols=1)
    plot_trace_vs_frequency(data_dict, value='real', max_mols=3)
    plot_trace_vs_frequency(data_dict, value='imag', max_mols=3)
    plot_matrix_elements_vs_freq(data_dict, max_mols=3)

    # If you want to attempt all molecules (lots of lines):
    # plot_trace_vs_frequency(data_dict, max_mols=None)
