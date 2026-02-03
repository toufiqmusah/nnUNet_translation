#!/usr/bin/env python3
"""
Example usage of the nn-diffusion framework Phase 1 implementation.
Demonstrates the core components without requiring full nnUNet installation.
"""

import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def example_1_ddpm_forward_process():
    """Example 1: Demonstrate DDPM forward diffusion process"""
    print("=" * 70)
    print("Example 1: DDPM Forward Diffusion Process")
    print("=" * 70)
    
    from nnunetv2.training.diffusion import DDPMStrategy
    
    # Create DDPM strategy with linear schedule
    strategy = DDPMStrategy(num_timesteps=1000, beta_schedule='linear')
    
    # Create a simple synthetic "image" (2D tensor)
    x0 = torch.randn(1, 1, 64, 64)
    
    # Show how noise is added at different timesteps
    timesteps = [0, 100, 250, 500, 750, 999]
    
    print(f"\nOriginal image shape: {x0.shape}")
    print(f"Testing diffusion at timesteps: {timesteps}\n")
    
    for t_val in timesteps:
        t = torch.tensor([t_val])
        noise = torch.randn_like(x0)
        xt = strategy.forward_process(x0, t, noise)
        
        # Compute signal-to-noise ratio
        signal_power = (x0 ** 2).mean()
        noise_power = ((xt - x0) ** 2).mean()
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
        
        print(f"t={t_val:4d}: SNR={snr.item():6.2f} dB, "
              f"mean={xt.mean().item():7.4f}, std={xt.std().item():6.4f}")
    
    print("\n✓ Forward process successfully adds noise progressively")


def example_2_noise_schedulers():
    """Example 2: Compare different noise schedules"""
    print("\n" + "=" * 70)
    print("Example 2: Comparing Noise Schedules")
    print("=" * 70)
    
    from nnunetv2.training.diffusion.noise_schedulers import (
        linear_beta_schedule,
        cosine_beta_schedule
    )
    
    num_timesteps = 1000
    
    # Get both schedules
    linear_betas = linear_beta_schedule(num_timesteps)
    cosine_betas = cosine_beta_schedule(num_timesteps)
    
    # Compute cumulative products
    linear_alphas_cumprod = torch.cumprod(1 - linear_betas, dim=0)
    cosine_alphas_cumprod = torch.cumprod(1 - cosine_betas, dim=0)
    
    print(f"\nLinear schedule:")
    print(f"  Beta range: [{linear_betas.min():.6f}, {linear_betas.max():.6f}]")
    print(f"  Final alpha_bar: {linear_alphas_cumprod[-1]:.6f}")
    
    print(f"\nCosine schedule:")
    print(f"  Beta range: [{cosine_betas.min():.6f}, {cosine_betas.max():.6f}]")
    print(f"  Final alpha_bar: {cosine_alphas_cumprod[-1]:.6f}")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(linear_betas.numpy(), label='Linear', alpha=0.7)
    ax1.plot(cosine_betas.numpy(), label='Cosine', alpha=0.7)
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Beta')
    ax1.set_title('Noise Schedule Comparison (Beta)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(linear_alphas_cumprod.numpy(), label='Linear', alpha=0.7)
    ax2.plot(cosine_alphas_cumprod.numpy(), label='Cosine', alpha=0.7)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Alpha_bar (cumulative product)')
    ax2.set_title('Signal Retention over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/noise_schedule_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Noise schedule comparison saved to /tmp/noise_schedule_comparison.png")


def example_3_time_embeddings():
    """Example 3: Demonstrate sinusoidal time embeddings"""
    print("\n" + "=" * 70)
    print("Example 3: Sinusoidal Time Embeddings")
    print("=" * 70)
    
    from nnunetv2.training.diffusion.utils import SinusoidalTimeEmbedding, TimeEmbeddingMLP
    
    # Create time embedder
    embed_dim = 128
    embedder = SinusoidalTimeEmbedding(embed_dim)
    mlp = TimeEmbeddingMLP(embed_dim)
    
    # Generate embeddings for different timesteps
    timesteps = torch.tensor([0, 100, 250, 500, 750, 999])
    embeddings = embedder(timesteps)
    processed_embeddings = mlp(embeddings)
    
    print(f"\nEmbedding dimension: {embed_dim}")
    print(f"Input timesteps: {timesteps.tolist()}")
    print(f"Output shape: {embeddings.shape}")
    print(f"\nEmbedding statistics:")
    print(f"  Mean: {embeddings.mean().item():.6f}")
    print(f"  Std:  {embeddings.std().item():.6f}")
    print(f"  Min:  {embeddings.min().item():.6f}")
    print(f"  Max:  {embeddings.max().item():.6f}")
    
    # Visualize embeddings
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.imshow(embeddings.T.numpy(), aspect='auto', cmap='RdBu', interpolation='nearest')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Embedding Dimension')
    ax1.set_title('Sinusoidal Time Embeddings')
    ax1.set_xticks(range(len(timesteps)))
    ax1.set_xticklabels(timesteps.tolist())
    plt.colorbar(ax1.images[0], ax=ax1, label='Value')
    
    ax2.imshow(processed_embeddings.T.detach().numpy(), aspect='auto', cmap='RdBu', interpolation='nearest')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Embedding Dimension')
    ax2.set_title('After MLP Processing')
    ax2.set_xticks(range(len(timesteps)))
    ax2.set_xticklabels(timesteps.tolist())
    plt.colorbar(ax2.images[0], ax=ax2, label='Value')
    
    plt.tight_layout()
    plt.savefig('/tmp/time_embeddings.png', dpi=150, bbox_inches='tight')
    print("\n✓ Time embeddings visualization saved to /tmp/time_embeddings.png")


def example_4_conditioning():
    """Example 4: Demonstrate conditioning methods"""
    print("\n" + "=" * 70)
    print("Example 4: Conditioning Methods")
    print("=" * 70)
    
    from nnunetv2.training.diffusion.conditioning import ConcatConditioning
    
    # Create conditioning method
    conditioning = ConcatConditioning()
    
    # Simulate noisy target and source image
    batch_size = 2
    noisy_target = torch.randn(batch_size, 3, 64, 64)
    source_image = torch.randn(batch_size, 3, 64, 64)
    
    print(f"\nNoisy target shape: {noisy_target.shape}")
    print(f"Source image shape: {source_image.shape}")
    
    # Prepare and apply conditioning
    cond = conditioning.prepare_conditioning(source_image)
    conditioned_input = conditioning.apply_conditioning(noisy_target, cond)
    
    print(f"Conditioned input shape: {conditioned_input.shape}")
    print(f"\n✓ Concatenation conditioning doubles the channel dimension")
    print("  This allows the model to see both the noisy target and source")


def example_5_training_target():
    """Example 5: Show what DDPM predicts during training"""
    print("\n" + "=" * 70)
    print("Example 5: DDPM Training Target (Noise Prediction)")
    print("=" * 70)
    
    from nnunetv2.training.diffusion import DDPMStrategy
    
    strategy = DDPMStrategy(num_timesteps=1000, beta_schedule='linear')
    
    # Simulate training scenario
    x0 = torch.randn(1, 1, 32, 32)
    t = torch.tensor([500])
    noise = torch.randn_like(x0)
    
    # Forward process
    xt = strategy.forward_process(x0, t, noise)
    
    # Get training target
    target = strategy.get_target(x0, noise, t)
    
    print(f"\nClean image (x0) shape: {x0.shape}")
    print(f"Noisy image (xt) shape: {xt.shape}")
    print(f"Training target shape: {target.shape}")
    print(f"\n✓ DDPM training target is the noise itself (epsilon prediction)")
    print("  Model learns to predict: epsilon = model(xt, t, conditioning)")
    
    # Verify target is the noise
    assert torch.allclose(target, noise), "Target should be the noise"
    print("  Verified: target == noise ✓")


def example_6_full_pipeline():
    """Example 6: Complete training step simulation"""
    print("\n" + "=" * 70)
    print("Example 6: Simulated Training Step")
    print("=" * 70)
    
    from nnunetv2.training.diffusion import DDPMStrategy
    from nnunetv2.training.diffusion.conditioning import ConcatConditioning
    from nnunetv2.training.diffusion.utils import SinusoidalTimeEmbedding, TimeEmbeddingMLP
    
    # Setup components
    strategy = DDPMStrategy(num_timesteps=1000, beta_schedule='cosine')
    conditioning_method = ConcatConditioning()
    time_embedder = SinusoidalTimeEmbedding(256)
    time_mlp = TimeEmbeddingMLP(256)
    
    # Simulate a batch
    batch_size = 4
    source = torch.randn(batch_size, 3, 64, 64)  # MR image
    target = torch.randn(batch_size, 3, 64, 64)  # CT image
    
    print(f"\nBatch size: {batch_size}")
    print(f"Source (e.g., MR) shape: {source.shape}")
    print(f"Target (e.g., CT) shape: {target.shape}")
    
    # Training step simulation
    print("\nTraining step:")
    
    # 1. Sample timesteps
    t = torch.randint(0, 1000, (batch_size,))
    print(f"  1. Sampled timesteps: {t.tolist()}")
    
    # 2. Sample noise
    noise = torch.randn_like(target)
    print(f"  2. Sampled noise with shape: {noise.shape}")
    
    # 3. Forward diffusion
    xt = strategy.forward_process(target, t, noise)
    print(f"  3. Applied forward diffusion: {xt.shape}")
    
    # 4. Prepare conditioning
    cond = conditioning_method.prepare_conditioning(source)
    model_input = conditioning_method.apply_conditioning(xt, cond)
    print(f"  4. Prepared model input: {model_input.shape}")
    
    # 5. Get time embeddings
    t_emb = time_embedder(t)
    t_emb = time_mlp(t_emb)
    print(f"  5. Generated time embeddings: {t_emb.shape}")
    
    # 6. Get training target
    training_target = strategy.get_target(target, noise, t)
    print(f"  6. Training target (noise): {training_target.shape}")
    
    # 7. Simulate model output and loss
    model_output = torch.randn_like(training_target)  # Simulated
    loss = strategy.compute_loss(model_output, training_target, t)
    print(f"  7. Computed loss: {loss.item():.6f}")
    
    print("\n✓ Complete training pipeline works end-to-end!")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("nn-diffusion Framework Phase 1 - Examples")
    print("=" * 70)
    print("\nThis demonstrates the core components without needing full nnUNet.")
    
    try:
        example_1_ddpm_forward_process()
        example_2_noise_schedulers()
        example_3_time_embeddings()
        example_4_conditioning()
        example_5_training_target()
        example_6_full_pipeline()
        
        print("\n" + "=" * 70)
        print("🎉 All examples completed successfully!")
        print("=" * 70)
        print("\nNext steps:")
        print("  - Phase 2: Network architecture modifications for time embeddings")
        print("  - Phase 2: DDIM implementation for faster sampling")
        print("  - Phase 2: Cross-attention conditioning")
        print("  - Phase 3: Flow matching and Brownian bridge strategies")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
