import torch
import math


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule from Nichol & Dhariwal (2021).
    Better for high-resolution images.
    """
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
