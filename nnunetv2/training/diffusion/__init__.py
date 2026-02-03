from .base_diffusion_strategy import DiffusionStrategy
from .schedulers.ddpm import DDPMStrategy
from .schedulers.ddim import DDIMStrategy

# Lazy import for trainer to avoid dependency issues
def __getattr__(name):
    if name == 'nnUNetDiffusionTrainer':
        from .base_diffusion_trainer import nnUNetDiffusionTrainer
        return nnUNetDiffusionTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'DiffusionStrategy',
    'nnUNetDiffusionTrainer', 
    'DDPMStrategy',
    'DDIMStrategy',
]
