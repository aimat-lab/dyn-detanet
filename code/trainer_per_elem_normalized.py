import torch
import wandb
from detanet_model import *
import utils as ut

# Import the scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from geomloss import SamplesLoss

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        loss_function=l2loss,
        device=torch.device("cuda"),
        optimizer='AdamW',
        lr=5e-7,            # Starting LR of 5e-4
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
        # Initialize Sinkhorn divergence (Earth Mover's Distance approximation)
        #self.sinkhorn_loss = SamplesLoss(loss="sinkhorn", backend="online")
        self.device = device

        self.opt_type = optimizer
        self.opts = {
            'AdamW': torch.optim.AdamW(self.model.parameters(), lr=lr, amsgrad=False),
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
        self.train_losses = []
        self.val_losses = []
        self.step = -1

    def train(
        self,
        num_train,
        targ,
        real_scaler = None,
        imag_scaler = None,
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
            running_emd_loss = 0.0
            num_batches = 0

            # We'll do a single "full" validation pass each epoch
            for j, batch in enumerate(self.train_data):
                self.step += 1
                torch.cuda.empty_cache()

                self.optimizer.zero_grad()

                if batch.x is not None:
                    out = self.model(
                        pos=batch.pos.to(self.device),
                        z=batch.z.to(self.device),
                        x_features=batch.x.to(self.device),
                        batch=batch.batch.to(self.device)
                    )
                else:
                    out = self.model(
                        pos=batch.pos.to(self.device),
                        z=batch.z.to(self.device),
                        batch=batch.batch.to(self.device)
                    )

                target = batch[targ].to(self.device)

                loss = self.loss_function(out.reshape(target.shape), target)
                wandb.log({"train_loss": loss.item(), 
                            "step": self.step})
                loss.backward()
                self.optimizer.step()
                running_train_loss += loss.item()
                num_batches += 1

                if epoch % 10 == 0:
                    # out shape [batch_size, 122, 3, 3]
                    # Renormalize the output
                    B = out.shape[0]
                    real_out = out[:, 61:, :, :] # [B, 61, 3, 3]
                    imag_out = out[:, :61, :, :] # [B, 61, 3, 3]
                    real_s = real_out.reshape(B * 61, 9) # [B * 61,9]
                    imag_s = imag_out.reshape(B * 61, 9) # [B * 61,9]

                    real_phys = torch.from_numpy(
                                real_scaler.inverse_transform(real_s.cpu().detach().numpy())
                            ).to(self.device)        # [B·61, 9]

                    imag_phys = torch.from_numpy(
                                imag_scaler.inverse_transform(imag_s.cpu().detach().numpy())
                            ).to(self.device)
                    

                    real_phys = real_phys.view(B, 61, 3, 3)
                    imag_phys = imag_phys.view(B, 61, 3, 3)
                    out_phys  = torch.cat([imag_phys, real_phys], dim=1)   # shape [B, 122, 3, 3]

                    # Renormalize the target
                    target = target.reshape(out.shape)
                    target_real = target[:, 61:, :, :] # [B, 61, 3, 3]
                    target_imag = target[:, :61, :, :] # [B, 61, 3, 3]
                    target_real_s = target_real.reshape(B * 61, 9) # [B * 61,9]
                    target_imag_s = target_imag.reshape(B* 61, 9) # [B * 61,9]
                    target_real_phys = torch.from_numpy(
                                real_scaler.inverse_transform(target_real_s.cpu().numpy())
                            ).to(self.device)
                    target_imag_phys = torch.from_numpy(
                                imag_scaler.inverse_transform(target_imag_s.cpu().numpy())
                            ).to(self.device)
                    

                    target_real_phys = target_real_phys.view(B, 61, 3, 3)
                    target_imag_phys = target_imag_phys.view(B, 61, 3, 3)
                    target_phys  = torch.cat([target_imag_phys, target_real_phys], dim=1)   # shape [B, 122, 3, 3]

                    # Compute the EMD loss
                    running_emd_loss += loss_emd_per_spectrum(out_phys, target_phys).item() #loss_emd(val_phys.reshape(val_target.shape), val_target)

                    #print(f"Batch {j+1}/{len_train}: ")
                    pass

            # Compute average training loss for epoch
            avg_train_loss = running_train_loss / num_batches
            self.train_losses.append(avg_train_loss)
            wandb.log({"epoch": epoch, "loss_per_epoch": avg_train_loss})


            if epoch % 10 == 0:
                self.emd_loss = running_emd_loss / num_batches
                wandb.log({
                    "emd_loss": self.emd_loss,
                    "epoch": epoch,
                })


            # Validation pass each epoch
            if self.val_data is not None:
                self.model.eval()
                running_val_loss_full = 0.0

                val_R2_v = 0.0
                val_count = 0
                val_emd_loss = 0.0

                with torch.no_grad():
                    for val_batch in self.val_data:
                        val_target = val_batch[targ].to(self.device)

                        if val_batch.x is not None:
                            val_out = self.model(
                                pos=val_batch.pos.to(self.device),
                                z=val_batch.z.to(self.device),
                                x_features=val_batch.x.to(self.device),
                                batch=val_batch.batch.to(self.device)
                            )
                        else:
                            val_out = self.model(
                                pos=val_batch.pos.to(self.device),
                                z=val_batch.z.to(self.device),
                                batch=val_batch.batch.to(self.device)
                            )

                        B = val_out.shape[0]

                        real_out = val_out[:, 61:, :, :] # [B, 61, 3, 3]
                        imag_out = val_out[:, :61, :, :] # [B, 61, 3, 3]

                        real_s = real_out.reshape(B * 61, 9) # [B * 61,9]
                        imag_s = imag_out.reshape(B * 61, 9) # [B * 61,9]

                        #real_phys = ut.inverse_std(real_s, real_scaler, device=self.device, )   # [B * 61,9]
                        #imag_phys = ut.inverse_std(imag_s, imag_scaler, device=self.device, )  # [B * 61,9]

                        # still on CPU / float64 → bring back to the right device afterwards
                        real_phys = torch.from_numpy(
                                    real_scaler.inverse_transform(real_s.cpu().numpy())
                                ).to(self.device)        # [B·61, 9]

                        imag_phys = torch.from_numpy(
                                    imag_scaler.inverse_transform(imag_s.cpu().numpy())
                                ).to(self.device)
                        

                        real_phys = real_phys.view(B, 61, 3, 3)     
                        imag_phys = imag_phys.view(B, 61, 3, 3)

                        val_phys  = torch.cat([imag_phys, real_phys], dim=1)   # shape [B, 122, 3, 3]                        
                        full_val_loss = self.loss_function(val_phys.reshape(val_target.shape), val_target).item()

                        running_val_loss_full += full_val_loss
                        val_count += 1

                        if epoch%10== 0:
                            val_R2_v += R2(val_phys.reshape(val_target.shape), val_target).item()

                            val_emd_loss += loss_emd_per_spectrum(val_phys, val_target.reshape(val_phys.shape)).item() #loss_emd(val_phys.reshape(val_target.shape), val_target)

                self.model.train()

                # Average val metrics
                avg_val_loss = running_val_loss_full / val_count

                # Log
                wandb.log({
                    "epoch": epoch,
                    "epoch_val_loss": avg_val_loss})


                if epoch%10== 0:
                    wandb.log({
                    "val_R2": val_R2_v / val_count,
                    "lr": self.optimizer.param_groups[0]['lr'],  # handy to see the LR
                    "val_emd_loss": val_emd_loss / val_count,
                    "epoch": epoch,
                    })

                self.val_losses.append(avg_val_loss)
                # Step the scheduler on the validation loss
                self.scheduler.step(avg_val_loss)


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
