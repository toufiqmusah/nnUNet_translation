import torch
import numpy as np
from typing import Optional, Callable, Union

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Define a simple passthrough if tqdm is not available
    def tqdm(iterable, **kwargs):
        return iterable


def ddpm_sample_loop(model: torch.nn.Module, 
                     diffusion_strategy,
                     shape: tuple,
                     conditioning: Optional[torch.Tensor] = None,
                     num_inference_steps: Optional[int] = None,
                     progress: bool = True,
                     return_intermediates: bool = False,
                     device: str = 'cuda') -> Union[torch.Tensor, tuple]:
    """
    Full sampling loop for DDPM.
    Generates samples by iteratively denoising from pure noise.
    
    Args:
        model: The denoising model
        diffusion_strategy: The DDPM/DDIM strategy instance
        shape: Shape of samples to generate (B, C, H, W)
        conditioning: Optional conditioning information
        num_inference_steps: Number of denoising steps (uses all timesteps if None)
        progress: Show progress bar
        return_intermediates: Return intermediate denoised samples
        device: Device to generate on
    
    Returns:
        samples: Generated samples [B, C, H, W]
        intermediates: List of intermediate samples (if return_intermediates=True)
    """
    model.eval()
    
    # Use all timesteps if not specified
    if num_inference_steps is None:
        num_inference_steps = diffusion_strategy.num_timesteps
    
    # Start from pure noise
    xt = torch.randn(shape, device=device)
    
    intermediates = [] if return_intermediates else None
    
    # Reverse diffusion process
    timesteps = range(diffusion_strategy.num_timesteps - 1, -1, -1)
    
    if progress:
        timesteps = tqdm(timesteps, desc="DDPM Sampling")
    
    with torch.no_grad():
        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # One denoising step
            xt = diffusion_strategy.sample_step(xt, t_batch, model, conditioning)
            
            if return_intermediates and t % (diffusion_strategy.num_timesteps // 10) == 0:
                intermediates.append(xt.cpu().clone())
    
    if return_intermediates:
        return xt, intermediates
    return xt


def ddim_sample_loop(model: torch.nn.Module,
                     diffusion_strategy,
                     shape: tuple,
                     conditioning: Optional[torch.Tensor] = None,
                     num_inference_steps: int = 50,
                     progress: bool = True,
                     return_intermediates: bool = False,
                     device: str = 'cuda') -> Union[torch.Tensor, tuple]:
    """
    Full sampling loop for DDIM with arbitrary timestep skipping.
    Much faster than DDPM by only using a subset of timesteps.
    
    Args:
        model: The denoising model
        diffusion_strategy: The DDIM strategy instance
        shape: Shape of samples to generate (B, C, H, W)
        conditioning: Optional conditioning information
        num_inference_steps: Number of denoising steps (can be much less than num_timesteps)
        progress: Show progress bar
        return_intermediates: Return intermediate denoised samples
        device: Device to generate on
    
    Returns:
        samples: Generated samples [B, C, H, W]
        intermediates: List of intermediate samples (if return_intermediates=True)
    """
    model.eval()
    
    # Create timestep schedule - skip timesteps uniformly
    # For example: if num_timesteps=1000 and num_inference_steps=50,
    # we use timesteps [999, 979, 959, ..., 19, 0] (every 20th step)
    total_timesteps = diffusion_strategy.num_timesteps
    skip = total_timesteps // num_inference_steps
    timesteps = list(range(total_timesteps - 1, -1, -skip))
    
    # Ensure we end at timestep 0
    if timesteps[-1] != 0:
        timesteps.append(0)
    
    # Start from pure noise
    xt = torch.randn(shape, device=device)
    
    intermediates = [] if return_intermediates else None
    
    if progress:
        timesteps_iter = tqdm(timesteps[:-1], desc=f"DDIM Sampling ({num_inference_steps} steps)")
    else:
        timesteps_iter = timesteps[:-1]
    
    with torch.no_grad():
        for i, t in enumerate(timesteps_iter):
            # Get previous timestep
            prev_t = timesteps[i + 1]
            
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            prev_t_batch = torch.full((shape[0],), prev_t, device=device, dtype=torch.long)
            
            # One denoising step (can skip multiple timesteps)
            xt = diffusion_strategy.sample_step_with_skip(xt, t_batch, prev_t_batch, 
                                                          model, conditioning)
            
            if return_intermediates and i % (len(timesteps) // 10) == 0:
                intermediates.append(xt.cpu().clone())
    
    if return_intermediates:
        return xt, intermediates
    return xt


def sample_with_strategy(model: torch.nn.Module,
                        diffusion_strategy,
                        shape: tuple,
                        strategy_name: str = 'ddpm',
                        conditioning: Optional[torch.Tensor] = None,
                        num_inference_steps: Optional[int] = None,
                        progress: bool = True,
                        return_intermediates: bool = False,
                        device: str = 'cuda') -> Union[torch.Tensor, tuple]:
    """
    Unified sampling interface that automatically selects the right sampling loop.
    
    Args:
        model: The denoising model
        diffusion_strategy: The diffusion strategy instance
        shape: Shape of samples to generate (B, C, H, W)
        strategy_name: Name of strategy ('ddpm' or 'ddim')
        conditioning: Optional conditioning information
        num_inference_steps: Number of inference steps
        progress: Show progress bar
        return_intermediates: Return intermediate samples
        device: Device to generate on
    
    Returns:
        samples: Generated samples
        intermediates: List of intermediate samples (if return_intermediates=True)
    """
    if strategy_name.lower() == 'ddim':
        if num_inference_steps is None:
            num_inference_steps = 50  # Default for DDIM
        return ddim_sample_loop(model, diffusion_strategy, shape, conditioning,
                               num_inference_steps, progress, return_intermediates, device)
    else:  # ddpm
        return ddpm_sample_loop(model, diffusion_strategy, shape, conditioning,
                               num_inference_steps, progress, return_intermediates, device)


def progressive_sampling(model: torch.nn.Module,
                        diffusion_strategy,
                        shape: tuple,
                        conditioning: Optional[torch.Tensor] = None,
                        save_frequency: int = 100,
                        device: str = 'cuda') -> list:
    """
    Sample and save intermediate results at regular intervals.
    Useful for visualization and understanding the denoising process.
    
    Args:
        model: The denoising model
        diffusion_strategy: The diffusion strategy
        shape: Shape of samples to generate
        conditioning: Optional conditioning
        save_frequency: Save every N timesteps
        device: Device to use
    
    Returns:
        samples_history: List of (timestep, sample) tuples
    """
    model.eval()
    
    xt = torch.randn(shape, device=device)
    samples_history = [(diffusion_strategy.num_timesteps, xt.cpu().clone())]
    
    with torch.no_grad():
        for t in range(diffusion_strategy.num_timesteps - 1, -1, -1):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            xt = diffusion_strategy.sample_step(xt, t_batch, model, conditioning)
            
            if t % save_frequency == 0 or t == 0:
                samples_history.append((t, xt.cpu().clone()))
    
    return samples_history


def interpolate_samples(model: torch.nn.Module,
                       diffusion_strategy,
                       x1: torch.Tensor,
                       x2: torch.Tensor,
                       num_steps: int = 10,
                       t_interpolate: int = 500,
                       conditioning: Optional[torch.Tensor] = None,
                       device: str = 'cuda') -> list:
    """
    Interpolate between two samples in the latent space.
    
    This adds noise to both samples up to timestep t_interpolate, then interpolates
    in the noisy latent space, and denoises back to samples.
    
    Args:
        model: The denoising model
        diffusion_strategy: The diffusion strategy
        x1: First sample
        x2: Second sample
        num_steps: Number of interpolation steps
        t_interpolate: Timestep to interpolate at (higher = more abstract)
        conditioning: Optional conditioning
        device: Device to use
    
    Returns:
        interpolated_samples: List of interpolated samples
    """
    model.eval()
    
    x1 = x1.to(device)
    x2 = x2.to(device)
    
    # Add noise to both samples
    t = torch.full((x1.shape[0],), t_interpolate, device=device, dtype=torch.long)
    noise1 = torch.randn_like(x1)
    noise2 = torch.randn_like(x2)
    
    xt1 = diffusion_strategy.forward_process(x1, t, noise1)
    xt2 = diffusion_strategy.forward_process(x2, t, noise2)
    
    # Interpolate in latent space
    alphas = torch.linspace(0, 1, num_steps, device=device)
    interpolated = []
    
    with torch.no_grad():
        for alpha in alphas:
            # Spherical linear interpolation (SLERP) for better interpolation
            xt_interp = (1 - alpha) * xt1 + alpha * xt2
            
            # Denoise from t_interpolate to 0
            xt = xt_interp
            for t_step in range(t_interpolate, -1, -1):
                t_batch = torch.full((x1.shape[0],), t_step, device=device, dtype=torch.long)
                xt = diffusion_strategy.sample_step(xt, t_batch, model, conditioning)
            
            interpolated.append(xt.cpu())
    
    return interpolated
