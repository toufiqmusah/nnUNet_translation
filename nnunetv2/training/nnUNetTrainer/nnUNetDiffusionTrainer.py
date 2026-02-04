import torch
from torch import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Union, Tuple, List
import numpy as np
import json
from time import time
from batchgenerators.utilities.file_and_folder_operations import join

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.diffusion.diffusion_strategy import DiffusionStrategy
from nnunetv2.training.diffusion.schedulers.ddpm import DDPMStrategy
from nnunetv2.training.diffusion.conditioning.concat_conditioning import ConcatConditioning
from nnunetv2.training.diffusion.utils.time_embedding import SinusoidalTimeEmbedding, TimeEmbeddingMLP
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


class nnUNetDiffusionTrainer(nnUNetTrainer):
    """
    Diffusion-based trainer for image-to-image translation.
    
    This trainer extends nnUNetTrainer to support diffusion models (DDPM, DDIM, Flow Matching, etc.)
    for medical image translation tasks like MR-to-CT synthesis.
    
    Key Architecture Details:
    - Network input: Concatenation of [noisy_target, source_image]
    - If source has N channels, network input has 2N channels
    - Output: Predicted noise (for DDPM) or other target based on strategy
    - Conditioning: Source image guides the denoising process
    
    Usage:
        nnUNetv2_train <dataset> 3d_fullres <fold> -tr nnUNetDiffusionTrainer -pl nnResUNetPlans
    
    Example:
        nnUNetv2_train 101 3d_fullres 0 -tr nnUNetDiffusionTrainer -pl nnResUNetPlans
    
    Key differences from standard nnUNetTrainer:
    - Uses diffusion forward/reverse processes
    - Incorporates time embeddings
    - Supports multiple diffusion strategies (DDPM, DDIM, etc.)
    - Conditioning on source images for image-to-image translation
    """
    
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        # Diffusion-specific settings (can be overridden)
        self.diffusion_strategy_name = 'ddpm'  # Default strategy
        self.num_timesteps = 1000
        self.beta_schedule = 'linear'
        
        # Will be initialized in initialize()
        self.diffusion_strategy = None
        self.time_embedder = None
        self.time_mlp = None
        self.conditioning_method = None
        
        # Override some defaults for diffusion training
        self.enable_deep_supervision = False
        self.num_epochs = 500  # Diffusion typically needs fewer epochs
        self.initial_lr = 1e-4  # Slightly lower LR for stability
    
    def initialize(self):
        """Initialize trainer with diffusion-specific components"""
        if not self.was_initialized:
            # Determine base number of input channels from dataset
            base_num_input_channels = determine_num_input_channels(
                self.plans_manager, 
                self.configuration_manager,
                self.dataset_json
            )
            
            # CRITICAL: For concatenation conditioning, we need to double the input channels
            # Network receives: concat([noisy_target, source_image], dim=1)
            # So if source has N channels, concatenated input has 2*N channels
            self.num_input_channels = base_num_input_channels * 2
            
            self.print_to_log_file(f"Base input channels: {base_num_input_channels}")
            self.print_to_log_file(f"Network input channels (after conditioning): {self.num_input_channels}")
            
            # Build network with doubled input channels
            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,  # This is now 2*base
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision,
            ).to(self.device)
            
            # Compile network if requested
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            # Configure optimizers
            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            
            # DDP wrapper if needed
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            # Build loss
            self.loss = self._build_loss()
            
            # Initialize diffusion strategy
            self.print_to_log_file(f"Initializing diffusion strategy: {self.diffusion_strategy_name}")
            self.diffusion_strategy = self._build_diffusion_strategy()
            
            # Move diffusion parameters to device
            self._move_diffusion_params_to_device()
            
            # Initialize time embeddings
            time_embed_dim = 256
            self.time_embedder = SinusoidalTimeEmbedding(time_embed_dim).to(self.device)
            self.time_mlp = TimeEmbeddingMLP(time_embed_dim).to(self.device)
            
            # Initialize conditioning
            self.conditioning_method = ConcatConditioning()
            
            self.print_to_log_file(f"Diffusion training initialized:")
            self.print_to_log_file(f"  Strategy: {self.diffusion_strategy_name}")
            self.print_to_log_file(f"  Timesteps: {self.num_timesteps}")
            self.print_to_log_file(f"  Beta schedule: {self.beta_schedule}")
            self.print_to_log_file(f"  Time embedding dim: {time_embed_dim}")
            self.print_to_log_file(f"  Conditioning: Concatenation (doubles input channels)")
            
            self.was_initialized = True
        else:
            raise RuntimeError("Trainer has already been initialized. If you need to re-initialize, please create a new trainer instance.")
    
    def _move_diffusion_params_to_device(self):
        """Move all diffusion strategy tensors to the correct device"""
        if hasattr(self.diffusion_strategy, 'betas'):
            self.diffusion_strategy.betas = self.diffusion_strategy.betas.to(self.device)
        if hasattr(self.diffusion_strategy, 'alphas'):
            self.diffusion_strategy.alphas = self.diffusion_strategy.alphas.to(self.device)
        if hasattr(self.diffusion_strategy, 'alphas_cumprod'):
            self.diffusion_strategy.alphas_cumprod = self.diffusion_strategy.alphas_cumprod.to(self.device)
        if hasattr(self.diffusion_strategy, 'alphas_cumprod_prev'):
            self.diffusion_strategy.alphas_cumprod_prev = self.diffusion_strategy.alphas_cumprod_prev.to(self.device)
        if hasattr(self.diffusion_strategy, 'sqrt_alphas_cumprod'):
            self.diffusion_strategy.sqrt_alphas_cumprod = self.diffusion_strategy.sqrt_alphas_cumprod.to(self.device)
        if hasattr(self.diffusion_strategy, 'sqrt_one_minus_alphas_cumprod'):
            self.diffusion_strategy.sqrt_one_minus_alphas_cumprod = self.diffusion_strategy.sqrt_one_minus_alphas_cumprod.to(self.device)
        if hasattr(self.diffusion_strategy, 'posterior_variance'):
            self.diffusion_strategy.posterior_variance = self.diffusion_strategy.posterior_variance.to(self.device)
        if hasattr(self.diffusion_strategy, 'posterior_log_variance_clipped'):
            self.diffusion_strategy.posterior_log_variance_clipped = self.diffusion_strategy.posterior_log_variance_clipped.to(self.device)
        if hasattr(self.diffusion_strategy, 'posterior_mean_coef1'):
            self.diffusion_strategy.posterior_mean_coef1 = self.diffusion_strategy.posterior_mean_coef1.to(self.device)
        if hasattr(self.diffusion_strategy, 'posterior_mean_coef2'):
            self.diffusion_strategy.posterior_mean_coef2 = self.diffusion_strategy.posterior_mean_coef2.to(self.device)
    
    def on_train_start(self):
        """Initialize training - called before first epoch"""
        # Call parent implementation
        super().on_train_start()
        
        # Initialize manual loss tracking (simpler than logger)
        self.train_losses = []
        self.validation_losses = []
        self._best_ema = None
        
        self.print_to_log_file("Diffusion training started")
        self.print_to_log_file(f"Training with {self.diffusion_strategy_name} strategy")
        self.print_to_log_file(f"Total epochs: {self.num_epochs}")
    
    def _build_loss(self):
        """
        Build loss for diffusion training.
        Loss is handled by the diffusion strategy, so we return a dummy loss here.
        """
        # The actual loss is computed in the diffusion strategy
        # We don't use the standard segmentation losses
        return None
    
    def _build_diffusion_strategy(self) -> DiffusionStrategy:
        """Build the diffusion strategy based on configuration"""
        if self.diffusion_strategy_name == 'ddpm':
            return DDPMStrategy(
                num_timesteps=self.num_timesteps,
                beta_schedule=self.beta_schedule
            )
        elif self.diffusion_strategy_name == 'ddim':
            from nnunetv2.training.diffusion.schedulers.ddim import DDIMStrategy
            return DDIMStrategy(
                num_timesteps=self.num_timesteps,
                beta_schedule=self.beta_schedule,
                eta=0.0  # Deterministic by default
            )
        else:
            raise ValueError(f"Unknown diffusion strategy: {self.diffusion_strategy_name}. "
                           f"Currently supported: ['ddpm', 'ddim']")
    
    def train_step(self, batch: dict) -> dict:
        """
        Diffusion training step.
        
        Process:
        1. Get source (condition) and target images
        2. Sample random timesteps
        3. Add noise to target (forward diffusion)
        4. Condition on source image via concatenation
        5. Predict noise (or other target based on strategy)
        6. Compute loss
        """
        data = batch['data']  # Source image (e.g., MR)
        target = batch['target']  # Target image (e.g., CT)
        
        # Move to device
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = target[0].to(self.device, non_blocking=True)
        else:
            target = target.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad(set_to_none=True)
        
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            batch_size = target.shape[0]
            
            # Sample random timesteps for each image in batch
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
            
            # Sample noise
            noise = torch.randn_like(target)
            
            # Forward diffusion: add noise to target
            x_t = self.diffusion_strategy.forward_process(target, t, noise)
            
            # Prepare conditioning from source
            conditioning = self.conditioning_method.prepare_conditioning(data)
            
            # Apply conditioning (concatenate source with noisy target)
            # Shape before: x_t [B, C, H, W, D], conditioning [B, C, H, W, D]
            # Shape after: model_input [B, 2C, H, W, D]
            model_input = self.conditioning_method.apply_conditioning(x_t, conditioning)
            
            # Debug: Uncomment to verify shapes
            # self.print_to_log_file(f"x_t shape: {x_t.shape}, conditioning shape: {conditioning.shape}, model_input shape: {model_input.shape}")
            
            # Get time embeddings (currently not used by network, will be added in Phase 2)
            # t_emb = self.time_embedder(t)
            # t_emb = self.time_mlp(t_emb)
            
            # Network forward pass
            # Note: Current network doesn't use time embeddings yet
            # This will be enhanced when we modify the U-Net architecture
            model_output = self.network(model_input)
            
            # Get training target (for DDPM, this is the noise)
            training_target = self.diffusion_strategy.get_target(target, noise, t)
            
            # Compute loss
            loss = self.diffusion_strategy.compute_loss(model_output, training_target, t)
        
        # Backward pass with gradient scaling
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
                batch_size = target.shape[0]
                t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
                
                noise = torch.randn_like(target)
                x_t = self.diffusion_strategy.forward_process(target, t, noise)
                
                conditioning = self.conditioning_method.prepare_conditioning(data)
                model_input = self.conditioning_method.apply_conditioning(x_t, conditioning)
                
                model_output = self.network(model_input)
                
                training_target = self.diffusion_strategy.get_target(target, noise, t)
                loss = self.diffusion_strategy.compute_loss(model_output, training_target, t)
        
        return {'loss': loss.detach().cpu().numpy()}
    
    def on_train_epoch_end(self, train_outputs: List[dict]):
        """Log training epoch statistics"""
        outputs_collated = collate_outputs(train_outputs)
        train_loss = np.mean(outputs_collated['loss'])
        
        self.train_losses.append(train_loss)
        
        self.print_to_log_file(f"Epoch {self.current_epoch} - Training loss: {train_loss:.6f}")
    
    def on_validation_epoch_end(self, val_outputs: List[dict]):
        """Handle validation epoch end"""
        outputs_collated = collate_outputs(val_outputs)
        val_loss = np.mean(outputs_collated['loss'])
        
        self.validation_losses.append(val_loss)
        
        self.print_to_log_file(f"Epoch {self.current_epoch} - Validation loss: {val_loss:.6f}")
        
        # Update EMA for checkpoint selection
        if self._best_ema is None:
            self._best_ema = val_loss
        else:
            self._best_ema = 0.9 * self._best_ema + 0.1 * val_loss
        
        self.print_to_log_file(f"Best EMA: {self._best_ema:.6f}")
    
    def on_epoch_end(self):
        """
        Called at the end of each epoch.
        Handle checkpointing and logging.
        """
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)
        
        # Save checkpoint periodically
        if (self.current_epoch + 1) % self.save_every == 0:
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))
        
        # Save best checkpoint if this is the best so far
        if self._best_ema is not None:
            current_val_loss = self.validation_losses[-1] if self.validation_losses else float('inf')
            if current_val_loss <= self._best_ema:
                self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))
                self.print_to_log_file(f"Saved new best checkpoint with val loss: {current_val_loss:.6f}")
        
        # Continue with standard nnUNet epoch end behavior
        self.print_to_log_file(f"Epoch {self.current_epoch} completed\n")
    
    def on_train_end(self):
        """Called when training finishes"""
        # Save loss history
        loss_dict = {
            'train_losses': self.train_losses,
            'validation_losses': self.validation_losses,
        }
        
        with open(join(self.output_folder, 'diffusion_loss_history.json'), 'w') as f:
            json.dump(loss_dict, f, indent=2)
        
        self.print_to_log_file("Training completed!")
        self.print_to_log_file(f"Final training loss: {self.train_losses[-1]:.6f}")
        self.print_to_log_file(f"Final validation loss: {self.validation_losses[-1]:.6f}")
        self.print_to_log_file(f"Best EMA: {self._best_ema:.6f}")
