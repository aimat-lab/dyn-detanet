import csv
import json

def transform_polarizabilities_auto_blocks(
    input_mol_csv_path: str,
    input_csv_path: str,
    output_csv_path: str
):
    with open(input_mol_csv_path, "r", encoding="utf-8") as infile_mol:
        reader = csv.reader(infile_mol)
        header = next(reader)
        smiles = []
        for row in reader:
            smiles.append(row[0])

    with open(input_csv_path, "r", encoding="utf-8") as infile:
        reader = csv.reader(infile)

        tensor_line = next(reader)       # e.g. ['#tensor','ee','ee','ee','em','em','em','mm','mm','mm']
        component_line = next(reader)    # e.g. ['#component','xx','xy','xz','xx','xy','xz','xx','xy','xz']
        freq_line = next(reader)         # e.g. ['#eV','1.55','1.55','1.55','2.00','2.00','2.00',...]

        tensor_entries = tensor_line[1:]
        component_entries = component_line[1:]
        freq_entries = freq_line[1:]

        tensor_entries = [t for t in tensor_entries if t.strip()]
        component_entries = [c for c in component_entries if c.strip()]
        freq_entries = [f for f in freq_entries if f.strip()]


        # Identify unique frequencies
        freq_values = list(map(float, freq_entries))  # convert to float
        unique_freqs = []
        for fv in freq_values:
            if fv not in unique_freqs:
                unique_freqs.append(fv)
        n_freqs = len(unique_freqs)


        # n_cols = n_freqs
        n_cols = len(freq_entries)
        blocks_info = []  # will hold dicts like {'name':'ee','size':6,'symmetry':True,'components':[...]}
        i = 0
        while i < n_cols:
            block_label = tensor_entries[i]  # e.g. 'ee'
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
                sym = True
            elif block_size == 9:
                sym = False
            else:
                sym = None  

            blocks_info.append({
                'label': block_label,
                'size': block_size,
                'symmetry': sym,
                'frequency': block_frequency,
                'start': block_start,
                'end': block_end
            })

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
            out_row = [smiles[mol_counter]]
            for binfo in blocks_info:
                bsize = binfo['size']
                block_start = binfo['start'] + 1 # +1 because the first entry of each row contains the mol id
                block_end = binfo['end'] + 1
                #print(f"blockstart = {block_start}; blockend = {block_end}")
                chunk = row[block_start : block_end]
                 # Build the 3×3 matrix from chunk
                if binfo['symmetry'] is True and len(chunk) == 6:
                    # e.g. chunk=[xx,xy,xz,yy,yz,zz]
                    xx, xy, xz, yy, yz, zz = chunk
                    mat_3x3 = [
                        [xx, xy, xz],
                        [xy, yy, yz],
                        [xz, yz, zz]
                    ]
                    out_row.append(json.dumps(mat_3x3))
                elif binfo['symmetry'] is False and len(chunk) == 9:
                    # e.g. chunk=[xx,xy,xz,yx,yy,yz,zx,zy,zz]
                    xx, xy, xz, yx, yy, yz, zx, zy, zz = chunk
                    mat_3x3 = [
                        [xx, xy, xz],
                        [yx, yy, yz],
                        [zx, zy, zz]
                    ]
                    out_row.append(json.dumps(mat_3x3))
                else:
                    print("What else is there?")
                    out_row.append(json.dumps(chunk))

            data_rows.append(out_row)
            mol_counter += 1

    # --- 5) Write the final CSV with all the data ---
    with open(output_csv_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(frequencies)
        writer.writerow(polarizability_type)
        writer.writerows(data_rows)

    print(f"Wrote {len(data_rows)} molecules to {output_csv_path}.")


if __name__ == "__main__":
    # Example usage
    transform_polarizabilities_auto_blocks(
        input_mol_csv_path = "/media/maria/work_space/capsule-3259363/data/HarvardOPV_40.csv",
        input_csv_path="/media/maria/work_space/capsule-3259363/data/all_polarizabilities.csv",
        output_csv_path="polarizabilities.csv"
    )
