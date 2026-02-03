import torch


def linear_beta_schedule(num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """Linear schedule from Ho et al. (2020)"""
    return torch.linspace(beta_start, beta_end, num_timesteps)
