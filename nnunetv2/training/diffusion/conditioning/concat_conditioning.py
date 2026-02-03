import torch
from .base_conditioning import ConditioningMethod


class ConcatConditioning(ConditioningMethod):
    """
    Simple channel-wise concatenation conditioning.
    Concatenates source image with noisy target along channel dimension.
    """
    
    def __init__(self):
        super().__init__()
    
    def prepare_conditioning(self, source: torch.Tensor) -> torch.Tensor:
        """Source image is used directly"""
        return source
    
    def apply_conditioning(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Concatenate along channel dimension"""
        return torch.cat([x, conditioning], dim=1)
