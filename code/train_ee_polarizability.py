import csv
import json
import os

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, 'data')

with open(data_dir + "/polarizabilities.csv", "r", encoding="utf-8") as infile:
    reader = csv.reader(infile)
    frequencies = next(reader)  # read column names
    pol_type = next(reader)

    print("pol_type", pol_type)
    
    for row in reader:
        # Suppose the matrix is in column 1 (just as an example):
        matrix_str = row[1]  # This cell has something like "[[\"257.294+1.120j\", \"...\"], [...], ...]"
        print(matrix_str)
        matrix_2d = json.loads(matrix_str)  # Now it's a list-of-lists of strings

        # Convert each string element to a Python complex:
        matrix_complex = [
            [complex(elem_str) for elem_str in row_list]
            for row_list in matrix_2d
        ]

        # Now matrix_complex is a 2D list of Python complex numbers
        print("Loaded 3×3 as complex:")
        for subrow in matrix_complex:
            print(subrow)

        # Use them (e.g. compute trace, do sums, etc.):
        trace_val = sum(matrix_complex[i][i] for i in range(3))
        print("Trace =", trace_val)

        # break after first row just for demo
        break