## Installation with Conda

### 1️⃣  Create and activate environment
conda create -n pyg_conda python=3.10 -y
conda activate pyg_conda

### 2️⃣  (CUDA 11.7, PyTorch 2.0)
conda install pytorch==2.0.0 pytorch-cuda==11.7 -c pytorch -c nvidia -y

### 3️⃣  Graph / materials libraries
conda install -c pyg          pyg>=2.5.2                -y          # PyTorch Geometric
conda install -c esri         torch-cluster>=1.6.3       -y
conda install -c conda-forge  torch-scatter>=2.1.1       -y
conda install -c conda-forge  nequip>=0.6.2 e3nn>=0.5.6  -y

### 4️⃣  Utilities & helpers
conda install -c conda-forge  pandas>=2.3.0 wandb>=0.20.1 -y
pip install "rdkit>=2025.3.3" "pot>=0.9.5"


## Installation one-liner

```
conda create -n detanet_env python=3.10 -y && \
conda activate detanet_env && \
conda install pytorch==2.0.0 pytorch-cuda==11.7 -c pytorch -c nvidia -y && \
conda install -c pyg pyg>=2.5.2 -y && \
conda install -c conda-forge nequip>=0.6.2 wandb>=0.20.1 torch-scatter>=2.1.1 pandas>=2.3.0 e3nn>=0.5.6 -y && \
conda install -c esri torch-cluster>=1.6.3 -y && \
pip install "rdkit>=2025.3.3" "pot>=0.9.5"
```

## Code Derived from
[DetaNet](https://codeocean.com/capsule/3259363/tree/v3)

## Cite:
### DetaNet:
Zou, Zihan & Zhang, Yujin & Liang, Lijun & Wei, Mingzhi & Leng, Jiancai & Jiang, Jun & Luo, Yi & Hu, Wei. (2023). A deep learning model for predicting selected organic molecular spectra. Nature Computational Science. 3. 1-8. 10.1038/s43588-023-00550-y. 

