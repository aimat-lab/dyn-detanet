import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyEmbedding(nn.Module):
    """
    A  MLP that maps a scalar frequency (omega) to a learned embedding of size 'embed_dim'.

    Example usage:
      freq_embed_module = FrequencyEmbedding(freq_in_dim=1, hidden_dim=16, embed_dim=32)
      # In forward pass:
      freq_in = torch.rand(batch_size)  # e.g. shape [batch]
      freq_embedding = freq_embed_module(freq_in)
      # freq_embedding shape = [batch, embed_dim], can be appended to GNN node features
    """
    def __init__(self, freq_in_dim=1, hidden_dim=16, embed_dim=32):
        super().__init__()
        self.linear1 = nn.Linear(freq_in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)


    def forward(self, freq):
        """
        freq : Tensor of shape [batch_size] or [batch_size, freq_in_dim]
        Returns a Tensor of shape [batch_size, embed_dim]
        """
        if freq.dim() == 1:
            # Convert from [batch_size] to [batch_size, 1]
            freq = freq.unsqueeze(-1)

        x = self.linear1(freq)    # shape: [batch_size, hidden_dim]
        x = F.silu(x)             # use SiLU (Swish) activation, or any other
        x = self.linear2(x)       # shape: [batch_size, embed_dim]
        return x
