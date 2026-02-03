from abc import ABC, abstractmethod
import torch


class DiffusionStrategy(ABC):
    """Abstract base class for diffusion training strategies"""
    
    def __init__(self, num_timesteps: int = 1000, **kwargs):
        self.num_timesteps = num_timesteps
    
    @abstractmethod
    def forward_process(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply forward diffusion process to add noise to clean data.
        
        Args:
            x0: Clean data [B, C, ...]
            t: Timesteps [B]
            noise: Sampled noise [B, C, ...]
        
        Returns:
            x_t: Noised data at timestep t
        """
        pass
    
    @abstractmethod
    def compute_loss(self, model_output: torch.Tensor, target: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute training loss for the diffusion model.
        
        Args:
            model_output: Model prediction
            target: Ground truth target
            t: Timesteps
        
        Returns:
            loss: Scalar loss value
        """
        pass
    
    @abstractmethod
    def sample_step(self, xt: torch.Tensor, t: torch.Tensor, model: torch.nn.Module, 
                   conditioning: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Perform one denoising step during sampling.
        
        Args:
            xt: Noisy data at timestep t
            t: Current timestep
            model: Denoising model
            conditioning: Optional conditioning information
        
        Returns:
            x_{t-1}: Denoised data at timestep t-1
        """
        pass
    
    @abstractmethod
    def get_target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Get the training target based on the strategy.
        
        Args:
            x0: Clean data
            noise: Sampled noise
            t: Timesteps
        
        Returns:
            target: Training target (e.g., noise for DDPM, velocity for flow matching)
        """
        pass
