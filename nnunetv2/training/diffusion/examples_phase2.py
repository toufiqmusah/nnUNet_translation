#!/usr/bin/env python3
"""
Example usage of Phase 2 features in the nn-diffusion framework.
Demonstrates DDIM, sigmoid schedule, and sampling utilities.
"""

import torch
import sys


def example_1_ddim_vs_ddpm():
    """Example 1: Compare DDIM and DDPM strategies"""
    print("=" * 70)
    print("Example 1: DDIM vs DDPM Comparison")
    print("=" * 70)
    
    from nnunetv2.training.diffusion import DDPMStrategy, DDIMStrategy
    
    # Create both strategies
    ddpm = DDPMStrategy(num_timesteps=1000, beta_schedule='linear')
    ddim = DDIMStrategy(num_timesteps=1000, beta_schedule='linear', eta=0.0)
    
    # Test forward process (same for both)
    x0 = torch.randn(1, 1, 64, 64)
    t = torch.tensor([500])
    noise = torch.randn_like(x0)
    
    xt_ddpm = ddpm.forward_process(x0, t, noise)
    xt_ddim = ddim.forward_process(x0, t, noise)
    
    print(f"\nForward process (should be identical):")
    print(f"  DDPM output: mean={xt_ddpm.mean():.6f}, std={xt_ddpm.std():.6f}")
    print(f"  DDIM output: mean={xt_ddim.mean():.6f}, std={xt_ddim.std():.6f}")
    print(f"  Difference: {(xt_ddpm - xt_ddim).abs().max():.10f}")
    
    print(f"\nKey differences:")
    print(f"  - DDPM: Stochastic sampling (adds noise at each step)")
    print(f"  - DDIM (eta=0): Deterministic sampling (no randomness)")
    print(f"  - DDIM can skip timesteps for faster inference")
    
    print("\n✓ Both strategies use same forward process")


def example_2_ddim_fast_sampling():
    """Example 2: Demonstrate DDIM's fast sampling via timestep skipping"""
    print("\n" + "=" * 70)
    print("Example 2: DDIM Fast Sampling")
    print("=" * 70)
    
    from nnunetv2.training.diffusion import DDIMStrategy
    
    strategy = DDIMStrategy(num_timesteps=1000, beta_schedule='linear', eta=0.0)
    
    # Simulate skipping from t=1000 to t=50 in fewer steps
    print(f"\nDDIM allows arbitrary timestep skipping:")
    print(f"  Total timesteps: {strategy.num_timesteps}")
    
    # Example: Use only 50 steps instead of 1000
    num_inference_steps = 50
    skip = strategy.num_timesteps // num_inference_steps
    
    print(f"  Inference steps: {num_inference_steps}")
    print(f"  Skip interval: {skip}")
    print(f"  Speed improvement: {strategy.num_timesteps / num_inference_steps:.1f}x faster")
    
    # Demonstrate a single skip
    xt = torch.randn(1, 1, 32, 32)
    t_current = torch.tensor([999])
    t_target = torch.tensor([950])
    
    class MockModel(torch.nn.Module):
        def forward(self, x, t, cond=None):
            return torch.randn_like(x) * 0.1  # Small noise for demo
    
    model = MockModel()
    x_prev = strategy.sample_step_with_skip(xt, t_current, t_target, model)
    
    print(f"\n  Example: Skip from t={t_current[0]} to t={t_target[0]}")
    print(f"  Input shape: {xt.shape}")
    print(f"  Output shape: {x_prev.shape}")
    
    print("\n✓ DDIM enables much faster sampling than DDPM")


def example_3_sigmoid_schedule():
    """Example 3: Compare noise schedules"""
    print("\n" + "=" * 70)
    print("Example 3: Sigmoid Noise Schedule")
    print("=" * 70)
    
    from nnunetv2.training.diffusion.noise_schedulers import (
        linear_beta_schedule,
        cosine_beta_schedule,
        sigmoid_beta_schedule
    )
    
    num_timesteps = 1000
    
    linear = linear_beta_schedule(num_timesteps)
    cosine = cosine_beta_schedule(num_timesteps)
    sigmoid = sigmoid_beta_schedule(num_timesteps)
    
    print(f"\nNoise schedule comparison (beta values):")
    print(f"  Linear:  [{linear.min():.6f}, {linear.max():.6f}]")
    print(f"  Cosine:  [{cosine.min():.6f}, {cosine.max():.6f}]")
    print(f"  Sigmoid: [{sigmoid.min():.6f}, {sigmoid.max():.6f}]")
    
    # Compare early, middle, late timesteps
    early_idx = 100
    mid_idx = 500
    late_idx = 900
    
    print(f"\nBeta values at different timesteps:")
    print(f"  t={early_idx:3d}: Linear={linear[early_idx]:.6f}, Cosine={cosine[early_idx]:.6f}, Sigmoid={sigmoid[early_idx]:.6f}")
    print(f"  t={mid_idx:3d}: Linear={linear[mid_idx]:.6f}, Cosine={cosine[mid_idx]:.6f}, Sigmoid={sigmoid[mid_idx]:.6f}")
    print(f"  t={late_idx:3d}: Linear={linear[late_idx]:.6f}, Cosine={cosine[late_idx]:.6f}, Sigmoid={sigmoid[late_idx]:.6f}")
    
    print(f"\nSigmoid schedule characteristics:")
    print(f"  - S-shaped curve (smooth transitions)")
    print(f"  - Gradual at beginning and end")
    print(f"  - Steeper in the middle")
    print(f"  - Can improve training stability")
    
    print("\n✓ Sigmoid schedule provides smoother noise progression")


def example_4_eta_control():
    """Example 4: DDIM eta parameter for stochasticity control"""
    print("\n" + "=" * 70)
    print("Example 4: DDIM Stochasticity Control (eta)")
    print("=" * 70)
    
    from nnunetv2.training.diffusion import DDIMStrategy
    
    # Create strategies with different eta values
    ddim_det = DDIMStrategy(num_timesteps=1000, eta=0.0)    # Fully deterministic
    ddim_semi = DDIMStrategy(num_timesteps=1000, eta=0.5)   # Semi-stochastic
    ddim_stoch = DDIMStrategy(num_timesteps=1000, eta=1.0)  # Fully stochastic (like DDPM)
    
    print(f"\nDDIM eta parameter controls stochasticity:")
    print(f"  eta = 0.0: Fully deterministic (default)")
    print(f"  eta = 0.5: Semi-stochastic (balanced)")
    print(f"  eta = 1.0: Fully stochastic (similar to DDPM)")
    
    print(f"\nCreated strategies:")
    print(f"  Deterministic: eta={ddim_det.eta}")
    print(f"  Semi-stochastic: eta={ddim_semi.eta}")
    print(f"  Stochastic: eta={ddim_stoch.eta}")
    
    print(f"\nUse cases:")
    print(f"  - eta=0.0: When you want reproducible results")
    print(f"  - eta>0.0: When you want sample diversity")
    print(f"  - eta=1.0: When you want DDPM-like quality with DDIM speed")
    
    print("\n✓ eta parameter provides flexible control over sampling")


def example_5_all_schedules_with_ddim():
    """Example 5: Use DDIM with all three schedules"""
    print("\n" + "=" * 70)
    print("Example 5: DDIM with Different Schedules")
    print("=" * 70)
    
    from nnunetv2.training.diffusion import DDIMStrategy
    
    schedules = ['linear', 'cosine', 'sigmoid']
    
    print(f"\nCreating DDIM strategies with different schedules:")
    
    for schedule in schedules:
        strategy = DDIMStrategy(num_timesteps=1000, beta_schedule=schedule, eta=0.0)
        
        # Test forward process
        x0 = torch.randn(1, 1, 32, 32)
        t = torch.tensor([500])
        noise = torch.randn_like(x0)
        xt = strategy.forward_process(x0, t, noise)
        
        print(f"  {schedule.capitalize():8s}: ✓ (output shape: {xt.shape})")
    
    print(f"\nAll schedules work with DDIM!")
    print(f"  - Choose based on your task and data")
    print(f"  - Cosine often works well for high-resolution images")
    print(f"  - Sigmoid provides smooth transitions")
    print(f"  - Linear is simple and stable")
    
    print("\n✓ DDIM is compatible with all noise schedules")


def example_6_inference_speed_comparison():
    """Example 6: Simulated speed comparison"""
    print("\n" + "=" * 70)
    print("Example 6: Inference Speed Comparison")
    print("=" * 70)
    
    num_timesteps = 1000
    
    # DDPM uses all timesteps
    ddpm_steps = num_timesteps
    
    # DDIM can use much fewer
    ddim_steps_options = [50, 100, 250]
    
    print(f"\nInference steps comparison:")
    print(f"  DDPM:      {ddpm_steps} steps (must use all)")
    
    for ddim_steps in ddim_steps_options:
        speedup = num_timesteps / ddim_steps
        print(f"  DDIM-{ddim_steps:3d}:  {ddim_steps} steps ({speedup:.1f}x faster)")
    
    print(f"\nTypical usage:")
    print(f"  - Training: Use DDPM or DDIM (same training)")
    print(f"  - Inference: Use DDIM with 50-250 steps")
    print(f"  - Quality/speed tradeoff: More steps = better quality")
    
    print(f"\nExample workflow:")
    print(f"  1. Train with DDPM (1000 timesteps)")
    print(f"  2. Inference with DDIM (50 steps) - 20x faster!")
    print(f"  3. Can use same trained model weights")
    
    print("\n✓ DDIM provides significant inference speedup")


def main():
    """Run all Phase 2 examples"""
    print("\n" + "=" * 70)
    print("nn-diffusion Framework Phase 2 - Examples")
    print("=" * 70)
    print("\nDemonstrating DDIM, sigmoid schedule, and sampling features")
    
    try:
        example_1_ddim_vs_ddpm()
        example_2_ddim_fast_sampling()
        example_3_sigmoid_schedule()
        example_4_eta_control()
        example_5_all_schedules_with_ddim()
        example_6_inference_speed_comparison()
        
        print("\n" + "=" * 70)
        print("🎉 All Phase 2 examples completed successfully!")
        print("=" * 70)
        print("\nKey takeaways:")
        print("  ✓ DDIM enables 10-20x faster inference than DDPM")
        print("  ✓ Sigmoid schedule provides smoother noise transitions")
        print("  ✓ eta parameter controls determinism vs diversity")
        print("  ✓ All schedules work with both DDPM and DDIM")
        print("  ✓ Training with DDPM, inference with DDIM is common")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
