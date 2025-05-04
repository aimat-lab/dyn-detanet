## Installation

conda create -n pyg_conda python=3.10

conda activate pyg_conda


conda install pytorch=2.0 pytorch-cuda=11.7 -c pytorch -c nvidia

conda install pyg -c pyg   

conda install conda-forge::nequip

conda install conda-forge::wandb 

conda install conda-forge::torch-scatter 

conda install esri::torch-cluster 

pip install rdkit

conda install pandas

To start a sweep: python code/spectra_pol_yaml.py --config polar_default.yaml --sweep