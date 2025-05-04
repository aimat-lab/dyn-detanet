import torch
import numpy as np

def l2loss(out,target):
    '''Mean Square Error(MSE)'''
    diff = out-target
    return torch.mean(diff ** 2)

def l1loss(out,target):
    '''Mean Absolute Error(MAE)'''
    return torch.mean(torch.abs(out-target))

def rmse(out, target):
    '''Root Mean Square Eroor(rmse) (also known as RMSD)'''
    return torch.sqrt(torch.mean((out - target) ** 2))

def state_l2loss(out,target):
    '''    #It is used to get the loss of the excited state vector,
    but we have tested that it does not perform well in QM9spectra's transition dipole prediction.


    from J. Phys. Chem. Lett. 2020, 11, 3828−3834
    <Combining SchNet and SHARC: The SchNarc Machine Learning Approach for Excited-State Dynamics>'''
    diffa=torch.abs(out-target)**2
    diffb=torch.abs(out+target)**2
    diff=torch.min(diffa,diffb)
    return torch.mean(diff)

def R2(out,target):
    '''coefficient of determination,Square of Pearson's coefficient, used to assess regression accuracy'''
    mean=torch.mean(target)
    SSE=torch.sum((out-target)**2)
    SST=torch.sum((mean-target)**2)
    return 1-(SSE/SST)

def combine_lose(out_tuple,target_tuple,lamb=10):
    '''Combine loss of energy and force,Training for ab initio dynamics simulation'''
    return l2loss(out_tuple[0],target_tuple[0])+lamb*l2loss(out_tuple[1],target_tuple[1])



def rf_rmse(pred, target, eps=1e-12):
    # pred, target : [batch, 62, 3, 3] complex
    diff2 = (pred - target).abs()**2        # |Δ|²
    num   = diff2.sum(dim=(-1, -2)).sum(dim=-1)          # Σ_f ||Δ||_F²
    den   = (target.abs()**2).sum(dim=(-1, -2)).sum(dim=-1)
    return torch.sqrt(num / (den + eps))    # [batch]

def cosine_similarity(pred: torch.Tensor,
                        target: torch.Tensor,
                        eps: float = 1e-12) -> torch.Tensor:
    """
    Complex cosine similarity  (one value per sample).

    cos_sim = |<x , y>| / (||x|| * ||y||)      in   [0 , 1]
              = 1 when x∥y   , 0 when orthogonal

    Parameters
    ----------
    pred, target : complex tensors
        Shape  [B, ...]  with the same remaining dimensions.
    eps : float
        Numerical safety term to avoid division by 0.

    Returns
    -------
    sims : real tensor  [B]
        cos-similarity for every item in the batch.
    """

    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")

    # flatten everything except the batch dimension  →  [B, N]
    x = pred.reshape(pred.shape[0], -1)
    y = target.reshape(target.shape[0], -1)
    inner = (x.conj() * y).sum(dim=-1) 
    nx = torch.linalg.norm(x, dim=-1)
    ny = torch.linalg.norm(y, dim=-1)
    cos_sim = inner.abs() / (nx * ny + eps)
    cos_loss = 1.0 - cos_sim      # still in [0,1]
    mean_loss = cos_loss.mean()   # scalar
    return mean_loss          # 1 → identical shape, 0 → orthogonal



import ot
def loss_emd(pred, target):

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
    B, R, C = pred.shape

    pred   = pred.reshape(B , R * C)    
    target = target.reshape(B, R * C)   
    losses = 0.0

    for i in range(R * C):
        x = pred[:, i]
        y = target[:, i]
        a = torch.full_like(x, 1 / len(x))      # all 1/B, non-negative
        b = torch.full_like(y, 1 / len(y))

        # Build [B, B] ground-distance matrix
        M = torch.cdist(x.unsqueeze(1), y.unsqueeze(1), p=2)
        #print("M", M.shape)
        #print("a", a.shape)
        #print("b", b.shape)

        a_np, b_np, M_np = map(torch.detach, (a, b, M))
        a_np, b_np, M_np = a_np.cpu().numpy(), b_np.cpu().numpy(), M_np.cpu().numpy()
        Wd      = ot.emd2(a_np, b_np, M_np, numItermax=2_000_000)            # exact Wasserstein

        losses += Wd            
    return losses / (R * C)  