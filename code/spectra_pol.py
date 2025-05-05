
import os
from torch_geometric.loader import DataLoader
from torch_geometric.loader import DataLoader

import trainer
from detanet_model import *

import wandb
import random
random.seed(42)

batch_size = 32
epochs = 100
lr=0.0006
cutoff=10
num_block=3
num_features=256
attention_head=16
scalar_outsize= (4* 62)#(4*62)
irreps_out= '124x2e'
out_type = 'cal_multi_tensor'
finetune = True
finetune_file = '/media/maria/work_space/dyn-detanet/code/trained_param/pretty-sweep_polar_Falsenormalize_70epochs_32bs_0_0006855550241449846lr_3blocks.pth' # "/home/maria/detanet_complex/code/trained_param/OPTpolar_70epochs_64batchsize_0.0009lr_6.0cutoff_6numblock_128features_onlyKITqm9_OPTIMIZED.pth"
target = 'y'
dataset_name = 'HOPV_dataset'

name = f"OPT_spectra_{epochs}epochs_{batch_size}batchsize_{lr}lr_{cutoff}cutoff_{num_block}numblock_{num_features}features_{dataset_name}"

# -------------------------------

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, 'data')

dataset = []
spec_data = []

# Load the dataset
dataset = torch.load(os.path.join(data_dir, dataset_name + '.pt'))
print(f"Number of graphs in the dataset: {len(dataset)}")

print(f"Total dataset length: {len(dataset)}")

ex1 = dataset[0]
ex2 = dataset[5]

print("dataset[0] :", ex1, )
print("dataset[5] :", ex2,)

# No normalization
for data in dataset:
    data.y = torch.cat([data.real, data.imag], dim=0)
    data.freq = data.freqs.repeat(len(data.z), 1)
    data.x = data.spectra.repeat(len(data.z), 1)

print("data.y.shape :", data.y.shape)



# -------------------------------
# Shuffle & Train/Val Split
# -------------------------------
random.shuffle(dataset)
train_frac = 0.9
split_index = int(train_frac * len(dataset))

train_datasets = dataset[:split_index]
val_datasets   = dataset[split_index:]

val_dataset_to_print = []
for mol in val_datasets:
    val_dataset_to_print.append(str(mol.idx))

print(f"Training set size: {len(train_datasets)}")
print(f"Validation set size: {len(val_datasets)}")

# Dataloaders
trainloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True,  drop_last=True)
valloader   = DataLoader(val_datasets,   batch_size=batch_size, shuffle=False, drop_last=True)


wandb.init(
    # set the wandb project where this run will be logged
    project="spectra-input",
    name=name,
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": "GNN",
    "dataset": dataset_name,
    "epochs": epochs,
    }
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = DetaNet(num_features=num_features,
                    act='swish',
                    maxl=3,
                    num_block=num_block, #3
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=attention_head,
                    rc=cutoff, #5.0
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=34,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=scalar_outsize, # 2,#4, 
                    irreps_out= irreps_out, #'2e',# '2e+2e',
                    summation=True,
                    norm=False,
                    out_type=out_type,
                    grad_type=None,
                    x_features=62,
                    device=device)
if finetune:
    state_dict=torch.load(finetune_file)
    model.load_state_dict(state_dict=state_dict)
    print("Model loaded from checkpoint")
model.train()
model.to(device)
wandb.watch(model, log="all")


trainer_ = trainer.Trainer(
    model,
    train_loader=trainloader,
    val_loader=valloader,
    loss_function=l2loss, #ut.fun_complex_multidimensional_loss, 
    lr=lr,
    weight_decay=0,
    optimizer='AdamW'
)

trainer_.train(num_train=epochs, targ=target)

torch.save(model.state_dict(), os.path.join(current_dir, 'trained_param', name + '.pth'))
