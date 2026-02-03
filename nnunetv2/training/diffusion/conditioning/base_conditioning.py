from abc import ABC, abstractmethod
import torch


class ConditioningMethod(ABC):
    """Abstract base class for conditioning methods"""
    
    @abstractmethod
    def prepare_conditioning(self, source: torch.Tensor) -> torch.Tensor:
        """Prepare conditioning from source image"""
        pass
    
    @abstractmethod
    def apply_conditioning(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """Apply conditioning to the input"""
        pass
