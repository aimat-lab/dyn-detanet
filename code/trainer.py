import torch
import wandb
from detanet_model import *

# Import the scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

        # -- ADD THE REDUCELRONPLATEAU SCHEDULER HERE --
        # min_lr=1e-5 ensures it won’t go below 1e-5
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.75,   # how much to reduce the LR by
            patience=5,   # how many epochs to wait before reducing
            min_lr=1e-10,  # do not reduce beyond 1e-10
            verbose=True
        )


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

            for j, batch in enumerate(self.train_data):
                self.step += 1
                torch.cuda.empty_cache()

                self.optimizer.zero_grad()

                out = self.model(
                    pos=batch.pos.to(self.device),
                    z=batch.z.to(self.device),
                    spectra=batch.spectra.to(self.device),
                    freqs=batch.freqs.to(self.device),
                    batch=batch.batch.to(self.device)
                )

                target = batch[targ].to(self.device)
                loss = self.loss_function(out.reshape(target.shape), target)
                wandb.log({"train_loss": loss.item(), 
                           "step": self.step})
                loss.backward()
                self.optimizer.step()

                # Accumulate epoch-level stats
                running_train_loss += loss.item()
                num_batches += 1

            # ------------------------------
            # End of one epoch
            # ------------------------------
            # Compute average training loss for epoch
            avg_train_loss = running_train_loss / num_batches
            self.train_losses.append(avg_train_loss)
            wandb.log({"epoch": epoch, "loss_per_epoch": avg_train_loss})

            # Validation pass
            if self.val_data is not None:
                self.model.eval()
                running_val_loss_full = 0.0
                val_mae = 0.0
                val_R2_v = 0.0
                val_count = 0
                with torch.no_grad():
                    for val_batch in self.val_data:
                        val_target = val_batch[targ].to(self.device)
                        val_out = self.model(
                            pos=val_batch.pos.to(self.device),
                            z=val_batch.z.to(self.device),
                            spectra=val_batch.spectra.to(self.device),
                            freqs=val_batch.freqs.to(self.device),
                            batch=val_batch.batch.to(self.device)
                        )
                        full_val_loss = self.loss_function(val_out.reshape(val_target.shape), val_target).item()
                        running_val_loss_full += full_val_loss
                        val_count += 1
                        val_mae += l1loss(val_out.reshape(val_target.shape), val_target).item()
                        val_R2_v += R2(val_out.reshape(val_target.shape), val_target).item()
                        wandb.log({"val_count": val_count})

                self.model.train()
                # Average val metrics
                avg_val_loss = running_val_loss_full / val_count
                avg_val_mae = val_mae / val_count
                avg_val_R = val_R2_v / val_count
                self.val_losses.append(avg_val_loss)

                # Step the scheduler on the validation loss
                self.scheduler.step(avg_val_loss)

                # Log
                wandb.log({
                    "epoch": epoch,
                    "epoch_val_loss": avg_val_loss,
                    "val_mae": avg_val_mae,
                    "val_R2": avg_val_R,
                    "lr": self.optimizer.param_groups[0]['lr'],  # handy to see the LR
                })

                print(f"Epoch {epoch+1}/{num_train}: "
                        f"Train loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}, "
                        f"Current LR={self.optimizer.param_groups[0]['lr']:.6e}")
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



