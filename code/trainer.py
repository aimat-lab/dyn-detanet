import copy
import os.path as osp
import csv
import util as ut
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import GRU, Linear, ReLU, Sequential

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import global_mean_pool
import torch_geometric
import logging
import os
from pathlib import Path
import json
import wandb
from detanet_model import *
import wandb
import logging

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        loss_function=l2loss,
        device=torch.device("cuda"),
        optimizer='AdamW',
        lr=5e-4,
        weight_decay=0
    ):
        """
        Args:
            model: Your PyTorch model
            train_loader: DataLoader for training set
            val_loader: DataLoader for validation set
            loss_function: Callable that takes (pred, target) => scalar loss
            device: CPU or GPU device
            optimizer: Name of optimizer to use
            lr: learning rate
            weight_decay: weight decay
        """
        self.model = model
        self.train_data = train_loader
        self.val_data = val_loader
        self.loss_function = loss_function
        self.device = device

        self.opt_type = optimizer
        self.opts = {
            'AdamW': torch.optim.AdamW(self.model.parameters(), lr=lr, amsgrad=False, weight_decay=weight_decay),
            'AdamW_amsgrad': torch.optim.AdamW(self.model.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay),
            'Adam': torch.optim.Adam(self.model.parameters(), lr=lr, amsgrad=False, weight_decay=weight_decay),
            'Adam_amsgrad': torch.optim.Adam(self.model.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay),
            'Adadelta': torch.optim.Adadelta(self.model.parameters(), lr=lr, weight_decay=weight_decay),
            'RMSprop': torch.optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay),
            'SGD': torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        }
        self.optimizer = self.opts[self.opt_type]

        # Step-based logs
        self.train_losses = []  # Train loss after each batch
        self.val_losses = []    # Val loss each time we do step-based validation

        self.step = -1

    def train(
        self,
        num_train,
        targ,
        stop_loss=1e-8,
        val_per_train=50,
        print_per_epoch=10
    ):
        """
        Args:
            num_train: Number of epochs
            targ: the name of the attribute in the batch for the target, e.g. 'y'
            stop_loss: If train/val loss drops below this, we stop early
            val_per_train: Do validation every N steps (mini-batches)
            print_per_epoch: Print logs every N steps
        """
        self.model.train()
        len_train = len(self.train_data)

        for epoch in range(num_train):
            running_train_loss = 0.0
            num_batches = 0

            # We'll do a single "full" validation pass each epoch, but we also do
            # step-based partial validation every val_per_train steps.
            running_val_loss_epoch = 0.0
            val_batches_in_epoch = 0

            # We re-init an iterator over val_data, for step-based val
            val_datas = iter(self.val_data) if self.val_data else None

            for j, batch in enumerate(self.train_data):
                self.step += 1
                torch.cuda.empty_cache()

                self.optimizer.zero_grad()

                out = self.model(
                    pos=batch.pos.to(self.device),
                    z=batch.z.to(self.device),
                    batch=batch.batch.to(self.device)
                )

                target = batch[targ].to(self.device)
                loss = self.loss_function(out.reshape(target.shape), target)
                loss.backward()
                self.optimizer.step()

                # Accumulate epoch-level stats
                running_train_loss += loss.item()
                num_batches += 1

            wandb.log({"train_loss": loss.item(), "epoch": epoch})
            # ------------------------------
            # End of one epoch
            # ------------------------------
            avg_train_loss = running_train_loss / num_batches
            self.train_losses.append(avg_train_loss)

            # Validation pass
            if self.val_data is not None:
                self.model.eval()
                running_val_loss_full = 0.0
                val_count = 0
                with torch.no_grad():
                    for val_batch in self.val_data:
                        val_target = val_batch[targ].to(self.device)
                        val_out = self.model(
                            pos=val_batch.pos.to(self.device),
                            z=val_batch.z.to(self.device),
                            batch=val_batch.batch.to(self.device)
                        )
                        full_val_loss = self.loss_function(val_out.reshape(val_target.shape), val_target).item()
                        running_val_loss_full += full_val_loss
                        val_count += 1
                        val_mae = l1loss(val_out.reshape(val_target.shape), val_target).item()
                        val_R2 = R2(val_out.reshape(val_target.shape), val_target).item()
                        wandb.log({
                            "val_loss": full_val_loss,
                            "val_mae": val_mae,
                            "val_R2": val_R2,
                            "epoch": epoch
                        })

                self.model.train()

                print("val_count", val_count)

                avg_val_loss = running_val_loss_full / val_count
                self.val_losses.append(avg_val_loss)
                wandb.log({"epoch": epoch, "epoch_val_loss": avg_val_loss})

                print(f"Epoch {epoch+1}/{num_train}: Train Lepoch_val_lossoss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{num_train}: Train Loss={avg_train_loss:.6f}")

    def load_state_and_optimizer(self, state_path=None, optimizer_path=None):
        if state_path is not None:
            state_dict = torch.load(state_path)
            self.model.load_state_dict(state_dict)
        if optimizer_path is not None:
            self.optimizer = torch.load(optimizer_path)

    def save_param(self, path):
        torch.save(self.model.state_dict(), path)

    def save_model(self, path):
        torch.save(self.model, path)

    def save_opt(self, path):
        torch.save(self.optimizer, path)

    def params(self):
        return self.model.state_dict()




"""
''' Trainer class that logs both train and validation losses. '''
class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        loss_function=l2loss,
        device=device,
        optimizer='Adam_amsgrad',
        lr=5e-4,
        weight_decay=0
    ):
        self.opt_type = optimizer
        self.device = device
        self.model = model
        self.train_data = train_loader
        self.val_data = val_loader
        self.device = device

        self.opts = {
            'AdamW': torch.optim.AdamW(self.model.parameters(), lr=lr, amsgrad=False, weight_decay=weight_decay),
            'AdamW_amsgrad': torch.optim.AdamW(self.model.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay),
            'Adam': torch.optim.Adam(self.model.parameters(), lr=lr, amsgrad=False, weight_decay=weight_decay),
            'Adam_amsgrad': torch.optim.Adam(self.model.parameters(), lr=lr, amsgrad=True, weight_decay=weight_decay),
            'Adadelta': torch.optim.Adadelta(self.model.parameters(), lr=lr, weight_decay=weight_decay),
            'RMSprop': torch.optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay),
            'SGD': torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        }
        self.optimizer = self.opts[self.opt_type]
        self.loss_function = loss_function

        self.step = -1

        # Lists to store loss values for plotting later
        self.train_losses = []
        self.val_losses = []

    def train(
        self,
        num_train,
        targ,
        stop_loss=1e-8,
        val_per_train=50,
        print_per_epoch=10
    ):
        self.model.train()
        len_train = len(self.train_data)

        for epoch in range(num_train):
            running_train_loss = 0.0
            num_batches = 0

            # If we want to do one validation pass per epoch
            running_val_loss = 0.0
            num_val_batches = 0
            val_datas = iter(self.val_data) if self.val_data else None

            for j, batch in enumerate(self.train_data):
                self.step += 1
                torch.cuda.empty_cache()

                self.optimizer.zero_grad()
                out = self.model(
                    pos=batch.pos.to(self.device),
                    z=batch.z.to(self.device),
                    batch=batch.batch.to(self.device)
                )

                target = batch[targ].to(self.device)
                loss = self.loss_function(out.reshape(target.shape), target)
                loss.backward()
                self.optimizer.step()
                
                
                running_train_loss += loss.item()
                num_batches += 1

                # Store train loss in the list
                self.train_losses.append(loss.item())

                # Log train loss to WandB
                wandb.log({"train_loss": loss.item(), "epoch": epoch})

                # Validation only every val_per_train steps
                if (self.step % val_per_train == 0) and (self.val_data is not None):
                    try:
                        val_batch = next(val_datas)
                    except StopIteration:
                        # If we exhaust the validation loader, reinitialize it
                        val_datas = iter(self.val_data)
                        val_batch = next(val_datas)

                    val_target = val_batch[targ].to(self.device)
                    self.model.eval()
                    val_out = self.model(
                        pos=val_batch.pos.to(self.device),
                        z=val_batch.z.to(self.device),
                        batch=val_batch.batch.to(self.device)
                    )

                    val_loss = self.loss_function(val_out.reshape(val_target.shape), val_target).item()
                    val_mae = l1loss(val_out.reshape(val_target.shape), val_target).item()
                    val_R2 = R2(val_out.reshape(val_target.shape), val_target).item()                

                    self.val_losses.append(val_loss)
                    self.model.train()

                    # Log these metrics to WandB
                    wandb.log({
                        "val_loss": val_loss,
                        "val_mae": val_mae,
                        "val_R2": val_R2,
                        "epoch": epoch
                    })

                    if self.step % print_per_epoch == 0:
                        logging.info(
                            f"Epoch[{self.step}/{num_train*len_train}], "
                            f"loss:{loss.item():.8f}, val_loss:{val_loss:.8f}, "
                            f"val_mae:{val_mae:.8f}, val_R2:{val_R2:.8f}"
                        )

                    # Early stop if loss is extremely low
                    assert (loss.item() > stop_loss) or (val_loss > stop_loss), (
                        "Training or validation loss < stop_loss. Stopping early."
                    )

                # If we only want to log training loss at certain intervals,
                # we can do so here (e.g., if (self.step % print_per_epoch == 0) ...).

    def load_state_and_optimizer(self, state_path=None, optimizer_path=None):
        if state_path is not None:
            state_dict = torch.load(state_path)
            self.model.load_state_dict(state_dict)
        if optimizer_path is not None:
            self.optimizer = torch.load(optimizer_path)

    def save_param(self, path):
        torch.save(self.model.state_dict(), path)

    def save_model(self, path):
        torch.save(self.model, path)

    def save_opt(self, path):
        torch.save(self.optimizer, path)

    def params(self):
        return self.model.state_dict()
"""