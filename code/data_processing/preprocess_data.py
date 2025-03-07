import csv
import json
import os

def load_polarizabilities(csv_path, pol_type='ee'):
    """
    Reads a CSV where:
      - Row 1 => frequencies
      - Row 2 => pol_type: 'ee','em','mm',...
      - Row 3+ => data rows:
         row[0] = SMILES (or ID)
         row[1..] = JSON strings of 3x3 complex polarizability, etc.

    We only extract columns with pol_type == 'ee'.
    Instead of storing a single complex 3×3, we split
    into real and imaginary parts => 'matrix_real' and 'matrix_imag'.

    Returns a list of dicts:
      {
        "smiles": ...,
        "frequency": ...,
        "polarizability_type": ...,
        "matrix_real": [[float, float, float], [..], [..]],
        "matrix_imag": [[float, float, float], [..], [..]]
      }
    """
    dataset = []

    with open(csv_path, "r", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        freq_line = next(reader)      # e.g. ["frequency", "1.55", "1.55", ...]
        pol_type_line = next(reader)  # e.g. ["polarizability_type","ee","em","mm","ee","em",...]

        # Identify which columns match the requested pol_type
        # skip col 0 because it might be "polartizability_type" or "smiles"
        selected_indices = []
        for col_idx, p_type in enumerate(pol_type_line):
            if p_type.strip().lower() == pol_type:
                selected_indices.append(col_idx)

        for row_i, row in enumerate(reader):
            if not row:
                continue

            smiles = row[0]
            # Ensure row has enough columns
            if selected_indices and len(row) <= max(selected_indices):
                # skip incomplete row
                continue

            for col_idx in selected_indices:
                # freq_line[col_idx] => frequency for that column
                freq_str = freq_line[col_idx]
                try:
                    freq_val = float(freq_str)
                except ValueError:
                    # If frequency is not numeric, skip
                    continue

                matrix_str = row[col_idx]
                if not matrix_str.strip():
                    # skip empty
                    continue

                # Parse JSON => nested list of strings like ["257.294+1.120j", ...]
                try:
                    matrix_2d = json.loads(matrix_str)
                except json.JSONDecodeError:
                    # skip malformed JSON
                    continue

                # Convert to 3×3 real+imag
                matrix_real = []
                matrix_imag = []
                for subrow in matrix_2d:
                    row_real = []
                    row_imag = []
                    for elem_str in subrow:
                        cval = complex(elem_str)  # parse "257.294+1.120j" -> Python complex
                        row_real.append(cval.real)
                        row_imag.append(cval.imag)
                    matrix_real.append(row_real)
                    matrix_imag.append(row_imag)

                data_entry = {
                    "smiles": smiles,
                    "frequency": freq_val,
                    "polarizability_type": pol_type,
                    "matrix_real": matrix_real,  # 3×3 of floats
                    "matrix_imag": matrix_imag   # 3×3 of floats
                }
                dataset.append(data_entry)

    return dataset


def save_dataset_to_csv(dataset, csv_path):
    """
    dataset is a list of dicts, each having keys:
      smiles, frequency, polarizability_type,
      matrix_real (3×3 float), matrix_imag (3×3 float)

    We'll write a CSV with columns:
      smiles, frequency, polarizability_type,
      matrix_real (JSON-encoded), matrix_imag (JSON-encoded)
    """
    with open(csv_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)

        # Write header
        writer.writerow([
            "smiles",
            "frequency",
            "polarizability_type",
            "matrix_real",
            "matrix_imag"
        ])

        for entry in dataset:
            smiles = entry["smiles"]
            freq_val = entry["frequency"]
            p_type = entry["polarizability_type"]
            mat_real = entry["matrix_real"]  # 3×3 of floats
            mat_imag = entry["matrix_imag"]  # 3×3 of floats

            # JSON-dump them so each becomes a single cell
            real_json = json.dumps(mat_real)
            imag_json = json.dumps(mat_imag)

            writer.writerow([smiles, freq_val, p_type, real_json, imag_json])

    print(f"Saved {len(dataset)} entries to {csv_path}.")


if __name__ == "__main__":
    current_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_dir = os.path.join(parent_dir, 'data')

    csv_input = os.path.join(data_dir, "polarizabilities.csv")
    csv_output = os.path.join(data_dir, "ee_polarizabilities.csv")

    data = load_polarizabilities(csv_input, pol_type='ee')
    save_dataset_to_csv(data, csv_output)
