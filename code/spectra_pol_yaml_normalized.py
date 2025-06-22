from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Any
import wandb  # deferred import so CI can skip easily


import yaml  # pip install pyyaml
import torch
import numpy as np
from torch_geometric.loader import DataLoader

# local imports
import trainer_per_elem_normalized
from detanet_model import DetaNet, l2loss  
from sklearn.preprocessing import StandardScaler


def build_dataset(data_file: Path, normalize: bool):
    dataset_file = Path(data_file)
    if not dataset_file.exists():
        raise FileNotFoundError(dataset_file)
    dataset = torch.load(dataset_file)
    print(f"Number of graphs in the dataset: {len(dataset)}")

    # Subtract static polarizability
    for data in dataset:
        data.real_ee = data.real_ee[1:]-data.real_ee[0]
        data.imag_ee = data.imag_ee[1:]

        data.y = torch.cat([data.real_ee, data.imag_ee], dim=0)
        data.x = data.spectra[1:].repeat(len(data.z), 1)
    return dataset


def split_dataset(dataset, train_fraction: float, seed: int):
    random.Random(seed).shuffle(dataset)
    idx = int(len(dataset) * train_fraction)
    return dataset[:idx], dataset[idx:]


def clip_by_value(matrix_2d: torch.Tensor,
                  low:  float = -2_500.0,
                  high: float =  2_500.0) -> torch.Tensor:
    """
    Return a copy of `matrix_2d` (shape [N, 9]) where every entry
    outside the interval [low, high] is set to the nearest bound.
    """
    return torch.clamp(matrix_2d, min=low, max=high)


def normalize_dataset(train_dataset):

    # Normalize the data (standard z-score)
    real_rows = torch.cat([d.real_ee.reshape(-1, 9) for d in train_dataset])  # [N·61, 9]
    imag_rows = torch.cat([d.imag_ee.reshape(-1, 9) for d in train_dataset])  # [N·61, 9]

    print("real_rows shape:", real_rows.shape)  # [N·61, 9]
    print("imag_rows shape:", imag_rows.shape)  # [N·61, 9]


    #robust_real = StandardScaler()     
    #R_scaled = robust_real.fit_transform(real_rows.numpy()).astype("float32")
    #robust_imag = StandardScaler()
    #I_scaled = robust_imag.fit_transform(imag_rows.numpy()).astype("float32")


    real_rows_clipped = clip_by_value(real_rows, -2_500, 2_500)
    imag_rows_clipped = clip_by_value(imag_rows, -2_500, 2_500)

    print("real_rows_clipped shape:", real_rows_clipped.shape)  # [N·61, 9]
    print("imag_rows_clipped shape:", imag_rows_clipped.shape)  # [N·61, 9]

    # 2) standard-scale the clipped data
    std_real = StandardScaler()
    std_imag = StandardScaler()

    R_clip = std_real.fit_transform(real_rows_clipped.numpy()).astype("float32")
    I_clip = std_imag.fit_transform(imag_rows_clipped.numpy()).astype("float32")


    offset = 0
    for d in train_dataset:
        n = d.real_ee.shape[0]                                # 61 frequency slices
        d.real_ee = torch.tensor(R_clip[offset:offset+n]).view(n, 3, 3)
        d.imag_ee = torch.tensor(I_clip[offset:offset+n]).view(n, 3, 3)
        offset += n
        d.y = torch.cat([d.real_ee, d.imag_ee], dim=0)  # [122, 3, 3]
        return train_dataset, std_real, std_imag


# -----------------------------------------------------------------------------
# Main training routine
# -----------------------------------------------------------------------------

def run_training(base_cfg: Dict[str, Any]):
    """Train DetaNet according to the hyper‑parameters in *cfg*."""

    # --- wandb ----------------------------------------------------------------
    wandb.init(
        project=base_cfg.get("project", "HOPV"),
        config=base_cfg,             # ← W&B merges sweep overrides here
    )

    cfg = dict(wandb.config)         # ← now cfg contains the sweep params



    # --- data -----------------------------------------------------------------
    dataset = build_dataset(cfg["data_file"], cfg["normalize"])
    train_ds, val_ds = split_dataset(dataset, cfg["train_fraction"], cfg["seed"])

    print(f"Training set size: {len(train_ds)}")
    print(f"Validation set size: {len(val_ds)}")
    # Print the indices of the validation set
    val_dataset_to_print = []
    for mol in val_ds:
        val_dataset_to_print.append(str(mol.idx))
    print("Validation set indices:", val_dataset_to_print)
    
    # --- normalize ------------------------------------------------------------
    train_ds, real_scaler, imag_scaler = normalize_dataset(train_ds)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True )
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False)


    name = (
        f"Clipped_norm_"   # ← single quotes here
        f"{cfg['epochs']}epochs_{cfg['batch_size']}bs_"
        f"{cfg['lr']}lr_{cfg['num_block']}blocks_{cfg['num_features']}features_{cfg['num_radial']}radial_"
        f"{cfg['attention_head']}heads_{cfg['cutoff']}cutoff_KITQM9"
    )
    print("Training name:", name)

    save_dir = Path("trained_param")
    save_dir.mkdir(exist_ok=True)
    fname = name + ".pt"
    print("Save dir:", save_dir)

    # --- device ---------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- model ----------------------------------------------------------------
    model = DetaNet(
        num_features       = cfg["num_features"],
        act                = "swish",
        maxl               = 3,
        num_block          = cfg["num_block"],
        radial_type        = "trainable_bessel",
        num_radial         = cfg["num_radial"],
        attention_head     = cfg["attention_head"],
        rc                 = cfg["cutoff"],
        dropout            = 0.0,
        use_cutoff         = False,
        max_atomic_number  = 34,
        atom_ref           = None,
        scale              = 1.0,
        scalar_outsize     = cfg["scalar_outsize"],
        irreps_out         = cfg["irreps_out"],
        summation          = True,
        norm               = False,
        out_type           = cfg["out_type"],
        grad_type          = None,
        x_features         = 62,
        device             = device,
    ).to(device)

    if cfg.get("finetune", False):
        ckpt = torch.load(cfg["finetune_file"], map_location=device)
        model.load_state_dict(ckpt)
        print("Loaded checkpoint", cfg["finetune_file"])

    wandb.watch(model, log="all")

    # --- trainer --------------------------------------------------------------
    trainer_ = trainer_per_elem_normalized.Trainer(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=l2loss, #ut.fun_complex_multidimensional_loss, 
        lr=cfg["lr"],
        weight_decay=0,
        optimizer='AdamW'
    )

    trainer_.train(num_train=cfg["epochs"], targ=cfg["target"], real_scaler=real_scaler, imag_scaler=imag_scaler)

    # --- save -----------------------------------------------------------------

    torch.save(model.state_dict(), save_dir / fname)
    print("Model saved to", save_dir / fname)


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train DetaNet with a YAML config file or run a W&B sweep.")
    parser.add_argument("--config", required=True, help="Path to YAML or JSON config file.")
    parser.add_argument("--sweep", action="store_true",
                        help="If set, start a random W&B sweep instead of a single run.")
    
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    if cfg_path.suffix in {".yaml", ".yml"}:
        cfg = yaml.safe_load(cfg_path.read_text())
    elif cfg_path.suffix == ".json":
        cfg = json.loads(cfg_path.read_text())
    else:
        raise ValueError("Config must be .yaml, .yml or .json")

    if args.sweep:
        #from sweep_utils import launch_random_sweep
        #launch_random_sweep(cfg_path,
        #                    project=cfg.get("project", "deta-random"),
        #                    trials=50)         # <-- change for bigger/smaller budgets
        from sweep_bayes_utils import launch_bayesian_sweep
        launch_bayesian_sweep(cfg_path,
                             project=cfg.get("project", "deta-bayes"),
                             trials=25)         # <-- change for bigger/smaller budgets       
        #wandb.agent(
         #   "uorcz-karlsruhe-institute-of-technology/polar-mm-KITQM9/7fao84ng",
          #  count=25)       
          # optional: how many runs to schedule
        
        return

    print("Yaml:", cfg)
    random.seed(cfg["seed"])
    run_training(cfg)


if __name__ == "__main__":
    main()
