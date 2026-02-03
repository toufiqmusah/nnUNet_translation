import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embeddings as used in Transformer and diffusion models.
    """
    
    def __init__(self, embedding_dim: int, max_period: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] tensor of timestep indices
        
        Returns:
            embeddings: [B, embedding_dim] tensor of sinusoidal embeddings
        """
        half_dim = self.embedding_dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if self.embedding_dim % 2 == 1:  # Zero pad if odd dimension
            emb = F.pad(emb, (0, 1))
        
        return emb


class TimeEmbeddingMLP(nn.Module):
    """
    MLP to process time embeddings, commonly used in diffusion models.
    """
    
    def __init__(self, time_embed_dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = time_embed_dim * 4
        
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, time_embed_dim),
        )
    
    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(t_emb)
