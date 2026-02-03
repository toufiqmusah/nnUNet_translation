"""
Diffusion framework for medical image translation.

This module provides the core diffusion components (strategies, schedulers, conditioning)
but the actual trainer is located in nnunetv2.training.nnUNetTrainer.nnUNetDiffusionTrainer
to follow nnUNet's trainer discovery convention.
"""

from .diffusion_strategy import DiffusionStrategy
from .schedulers.ddpm import DDPMStrategy
from .schedulers.ddim import DDIMStrategy

__all__ = [
    'DiffusionStrategy',
    'DDPMStrategy',
    'DDIMStrategy',
]
