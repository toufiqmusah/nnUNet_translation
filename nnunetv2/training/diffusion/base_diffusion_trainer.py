import torch
from torch import autocast
from typing import Union, Tuple, List
import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from .base_diffusion_strategy import DiffusionStrategy
from .schedulers.ddpm import DDPMStrategy
from .conditioning.concat_conditioning import ConcatConditioning
from .utils.time_embedding import SinusoidalTimeEmbedding, TimeEmbeddingMLP
from nnunetv2.utilities.helpers import dummy_context


class nnUNetDiffusionTrainer(nnUNetTrainer):
    """
    Base trainer for diffusion-based image-to-image translation.
    Extends nnUNetTrainer with diffusion-specific training logic.
    """
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
        diffusion_strategy: str = 'ddpm',
        num_timesteps: int = 1000,
        beta_schedule: str = 'linear',
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        # Diffusion-specific settings
        self.diffusion_strategy_name = diffusion_strategy
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        
        # Initialize diffusion components (will be set in initialize())
        self.diffusion_strategy = None
        self.time_embedder = None
        self.conditioning_method = None
        
        # Override some nnUNet defaults for diffusion
        self.enable_deep_supervision = False  # Typically not used in diffusion
        self.num_epochs = 500  # Diffusion may need fewer epochs
    
    def initialize(self):
        """Override to add diffusion-specific initialization"""
        super().initialize()
        
        # Initialize diffusion strategy
        self.diffusion_strategy = self._build_diffusion_strategy()
        
        # Initialize time embedding
        time_embed_dim = 256  # Standard choice
        self.time_embedder = SinusoidalTimeEmbedding(time_embed_dim).to(self.device)
        self.time_mlp = TimeEmbeddingMLP(time_embed_dim).to(self.device)
        
        # Initialize conditioning method
        self.conditioning_method = ConcatConditioning()
        
        self.print_to_log_file(f"Initialized diffusion training with {self.diffusion_strategy_name} strategy")
        self.print_to_log_file(f"Num timesteps: {self.num_timesteps}, Beta schedule: {self.beta_schedule}")
    
    def _build_diffusion_strategy(self) -> DiffusionStrategy:
        """Build the specified diffusion strategy"""
        if self.diffusion_strategy_name == 'ddpm':
            return DDPMStrategy(
                num_timesteps=self.num_timesteps,
                beta_schedule=self.beta_schedule
            )
        elif self.diffusion_strategy_name == 'ddim':
            from .schedulers.ddim import DDIMStrategy
            return DDIMStrategy(
                num_timesteps=self.num_timesteps,
                beta_schedule=self.beta_schedule,
                eta=0.0  # Deterministic by default
            )
        else:
            raise ValueError(f"Unknown diffusion strategy: {self.diffusion_strategy_name}")
    
    def train_step(self, batch: dict) -> dict:
        """Override training step for diffusion training"""
        data = batch['data']  # Source image (e.g., MR)
        target = batch['target']  # Target image (e.g., CT)
        
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = target[0].to(self.device, non_blocking=True)
        else:
            target = target.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # Sample random timesteps
            batch_size = target.shape[0]
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
            
            # Sample noise
            noise = torch.randn_like(target)
            
            # Forward diffusion process
            x_t = self.diffusion_strategy.forward_process(target, t, noise)
            
            # Prepare conditioning from source image
            conditioning = self.conditioning_method.prepare_conditioning(data)
            
            # Apply conditioning (e.g., concatenate)
            model_input = self.conditioning_method.apply_conditioning(x_t, conditioning)
            
            # Get time embeddings
            t_emb = self.time_embedder(t)
            t_emb = self.time_mlp(t_emb)
            
            # Model prediction (network needs to be modified to accept time embeddings)
            # For now, pass through network (will be enhanced in later phases)
            model_output = self.network(model_input)
            
            # Get target based on strategy
            training_target = self.diffusion_strategy.get_target(target, noise, t)
            
            # Compute loss
            loss = self.diffusion_strategy.compute_loss(model_output, training_target, t)
        
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()
        
        return {'loss': loss.detach().cpu().numpy()}
    
    def validation_step(self, batch: dict) -> dict:
        """Validation step for diffusion models"""
        data = batch['data']
        target = batch['target']
        
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = target[0].to(self.device, non_blocking=True)
        else:
            target = target.to(self.device, non_blocking=True)
        
        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                # Sample random timesteps
                batch_size = target.shape[0]
                t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
                
                noise = torch.randn_like(target)
                x_t = self.diffusion_strategy.forward_process(target, t, noise)
                
                conditioning = self.conditioning_method.prepare_conditioning(data)
                model_input = self.conditioning_method.apply_conditioning(x_t, conditioning)
                
                t_emb = self.time_embedder(t)
                t_emb = self.time_mlp(t_emb)
                
                model_output = self.network(model_input)
                
                training_target = self.diffusion_strategy.get_target(target, noise, t)
                loss = self.diffusion_strategy.compute_loss(model_output, training_target, t)
        
        return {'loss': loss.detach().cpu().numpy()}
