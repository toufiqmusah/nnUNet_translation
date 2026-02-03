import torch
import math


def sigmoid_beta_schedule(num_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02, 
                          sig_range: float = 6.0) -> torch.Tensor:
    """
    Sigmoid schedule for beta values.
    Provides smoother transitions between low and high noise levels compared to linear schedule.
    
    Args:
        num_timesteps: Number of diffusion timesteps
        beta_start: Minimum beta value
        beta_end: Maximum beta value
        sig_range: Range of sigmoid function (default: 6.0 gives smooth S-curve)
    
    Returns:
        betas: Tensor of beta values [num_timesteps]
    
    Reference: 
    The sigmoid function provides a smooth S-curve that can be more gradual at the 
    beginning and end of the diffusion process, which can help with training stability.
    """
    # Create sigmoid curve from -sig_range/2 to sig_range/2
    steps = num_timesteps
    t = torch.linspace(-sig_range / 2, sig_range / 2, steps)
    
    # Apply sigmoid function: sigmoid(t) = 1 / (1 + exp(-t))
    sigmoid_values = torch.sigmoid(t)
    
    # Normalize to [0, 1] range
    sigmoid_normalized = (sigmoid_values - sigmoid_values[0]) / (sigmoid_values[-1] - sigmoid_values[0])
    
    # Scale to [beta_start, beta_end] range
    betas = beta_start + sigmoid_normalized * (beta_end - beta_start)
    
    return betas
