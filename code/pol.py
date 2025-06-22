
import os
from torch_geometric.loader import DataLoader
import torch_geometric

import trainer
from detanet_model import *

import wandb
import random
random.seed(42)

batch_size = 32
epochs = 50
lr=0.0005 
cutoff=6.0
num_block=6
num_features=256


scalar_outsize= 4
irreps_out= '2x2e'
out_type = 'multi_tensor'

finetune = False
finetune_file = "/home/maria/detanet_complex/code/trained_param/OPTpolar_70epochs_64batchsize_0.0009lr_6.0cutoff_6numblock_128features_onlyKITqm9_OPTIMIZED.pth"

target = 'y'
x_features = 64
dataset_name = 'KITQM9'

name = f"Per_elem_{epochs}epochs_{batch_size}batchsize_{lr}lr_{cutoff}cutoff_{num_block}numblock_{num_features}features_{dataset_name}"

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

ref_dataset = []

# No normalization
for data in dataset:
    for i in range(data.real_ee.shape[0]):
        y = torch.cat([data.real_ee[i].unsqueeze(0), data.imag_ee[i].unsqueeze(0)], dim=0)  # -> [2, 3, 3]
        freq = data.freqs[i].unsqueeze(0)
        spec_freq = data.spectra[i].unsqueeze(0)
        #x = torch.cat([data.osc_pos, data.osc_strength], dim= 0)
        #x = torch.cat([freq, x], dim=0)
        x = torch.cat([freq, spec_freq, data.spectra], dim=0)

        # Create the dataset entry
        data_entry = torch_geometric.data.Data(
            idx=data.idx,
            pos=data.pos,
            z=torch.LongTensor(data.z),
            x=x.repeat(len(data.z), 1),
            y=y      
        )
        ref_dataset.append(data_entry)


print(f"Number of graphs in the dataset: {len(ref_dataset)}")

print(ref_dataset[0])
# -------------------------------
# Shuffle & Train/Val Split
# -------------------------------

# Train and validate per molecule
unique_mol_ids = list({data_entry.idx for data_entry in ref_dataset})
random.shuffle(unique_mol_ids)

num_val_mols = max(1, int(0.1 * len(unique_mol_ids)))  # e.g., 10% for validation
val_mol_ids = set(unique_mol_ids[:num_val_mols])

train_datasets = [d for d in ref_dataset if d.idx not in val_mol_ids]
val_datasets = [d for d in ref_dataset if d.idx in val_mol_ids]

print(f"Total unique molecules: {len(unique_mol_ids)}")
print(f"Validation molecule IDs: {sorted(val_mol_ids)}")
print(f"Training set size: {len(train_datasets)}")
print(f"Validation set size: {len(val_datasets)}")


'''Using torch_Geometric.dataloader.DataLoader Converts a dataset into a batch of 64 molecules of training data.'''

trainloader=DataLoader(train_datasets,batch_size=batch_size,shuffle=True, drop_last=True)
valloader=DataLoader(val_datasets,batch_size=batch_size,shuffle=True, drop_last=True)



wandb.init(
    # set the wandb project where this run will be logged
    project="OPT-configs",
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

model = DetaNet(num_features=128,
                    act='swish',
                    maxl=3,
                    num_block=num_block,
                    radial_type='trainable_bessel',
                    num_radial=32,
                    attention_head=8,
                    rc=cutoff,
                    dropout=0.0,
                    use_cutoff=False,
                    max_atomic_number=34,
                    atom_ref=None,
                    scale=1.0,
                    scalar_outsize=scalar_outsize,
                    irreps_out= irreps_out, 
                    summation=True,
                    norm=False,
                    out_type=out_type,
                    grad_type=None,
                    x_features=x_features,
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
