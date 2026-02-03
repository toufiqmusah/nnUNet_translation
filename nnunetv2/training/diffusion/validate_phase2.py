#!/usr/bin/env python3
"""
Validation script for Phase 2 of the nn-diffusion framework.
Tests DDIM strategy, sigmoid schedule, and sampling utilities.
"""

import sys
import torch
import numpy as np


def test_imports():
    """Test that all Phase 2 modules can be imported"""
    print("Testing Phase 2 imports...")
    try:
        from nnunetv2.training.diffusion import (
            DiffusionStrategy,
            DDPMStrategy,
            DDIMStrategy
        )
        from nnunetv2.training.diffusion.noise_schedulers import (
            linear_beta_schedule,
            cosine_beta_schedule,
            sigmoid_beta_schedule
        )
        from nnunetv2.training.diffusion.utils import (
            ddpm_sample_loop,
            ddim_sample_loop,
            sample_with_strategy,
            progressive_sampling,
            interpolate_samples
        )
        print("✓ All Phase 2 imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ddim_forward_process():
    """Test DDIM forward process (same as DDPM)"""
    print("\nTesting DDIM forward process...")
    try:
        from nnunetv2.training.diffusion import DDIMStrategy
        
        # Create DDIM strategy
        strategy = DDIMStrategy(num_timesteps=1000, beta_schedule='linear', eta=0.0)
        
        # Create test data
        batch_size = 2
        x0 = torch.randn(batch_size, 3, 32, 32)
        t = torch.randint(0, 1000, (batch_size,))
        noise = torch.randn_like(x0)
        
        # Test forward process
        xt = strategy.forward_process(x0, t, noise)
        
        assert xt.shape == x0.shape, "Output shape mismatch"
        assert not torch.isnan(xt).any(), "NaN values in output"
        
        print(f"  Input shape: {x0.shape}")
        print(f"  Output shape: {xt.shape}")
        print(f"  Timesteps: {t.tolist()}")
        print("✓ DDIM forward process works correctly")
        return True
    except Exception as e:
        print(f"✗ DDIM forward process failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ddim_sample_step():
    """Test DDIM sampling step"""
    print("\nTesting DDIM sampling step...")
    try:
        from nnunetv2.training.diffusion import DDIMStrategy
        
        # Create DDIM strategy (deterministic)
        strategy = DDIMStrategy(num_timesteps=1000, beta_schedule='linear', eta=0.0)
        
        # Create test data
        xt = torch.randn(2, 3, 32, 32)
        t = torch.tensor([500, 500])
        
        # Mock model that returns random noise prediction
        class MockModel(torch.nn.Module):
            def forward(self, x, t, cond=None):
                return torch.randn_like(x)
        
        model = MockModel()
        
        # Test sampling step
        x_prev = strategy.sample_step(xt, t, model)
        
        assert x_prev.shape == xt.shape, "Output shape mismatch"
        assert not torch.isnan(x_prev).any(), "NaN in output"
        
        print(f"  Input shape: {xt.shape}")
        print(f"  Output shape: {x_prev.shape}")
        print("✓ DDIM sampling step works correctly")
        return True
    except Exception as e:
        print(f"✗ DDIM sampling step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ddim_skip_sampling():
    """Test DDIM sampling with timestep skipping"""
    print("\nTesting DDIM skip sampling...")
    try:
        from nnunetv2.training.diffusion import DDIMStrategy
        
        strategy = DDIMStrategy(num_timesteps=1000, beta_schedule='linear', eta=0.0)
        
        # Test skipping from t=999 to t=500
        xt = torch.randn(2, 3, 32, 32)
        t = torch.tensor([999, 999])
        prev_t = torch.tensor([500, 500])
        
        class MockModel(torch.nn.Module):
            def forward(self, x, t, cond=None):
                return torch.randn_like(x)
        
        model = MockModel()
        
        # Test sampling with skip
        x_prev = strategy.sample_step_with_skip(xt, t, prev_t, model)
        
        assert x_prev.shape == xt.shape, "Output shape mismatch"
        assert not torch.isnan(x_prev).any(), "NaN in output"
        
        print(f"  Skipped from t={t[0].item()} to t={prev_t[0].item()}")
        print(f"  Output shape: {x_prev.shape}")
        print("✓ DDIM skip sampling works correctly")
        return True
    except Exception as e:
        print(f"✗ DDIM skip sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sigmoid_schedule():
    """Test sigmoid noise schedule"""
    print("\nTesting sigmoid noise schedule...")
    try:
        from nnunetv2.training.diffusion.noise_schedulers import sigmoid_beta_schedule
        
        num_timesteps = 1000
        betas = sigmoid_beta_schedule(num_timesteps)
        
        assert betas.shape == (num_timesteps,), "Shape mismatch"
        assert (betas >= 0).all() and (betas <= 1).all(), "Betas out of range"
        assert betas[0] < betas[-1], "Sigmoid schedule should increase"
        
        print(f"  Schedule length: {len(betas)}")
        print(f"  Beta range: [{betas.min():.6f}, {betas.max():.6f}]")
        print(f"  First 3: {betas[:3].tolist()}")
        print(f"  Last 3: {betas[-3:].tolist()}")
        print("✓ Sigmoid schedule works correctly")
        return True
    except Exception as e:
        print(f"✗ Sigmoid schedule failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ddim_with_sigmoid():
    """Test DDIM with sigmoid schedule"""
    print("\nTesting DDIM with sigmoid schedule...")
    try:
        from nnunetv2.training.diffusion import DDIMStrategy
        
        strategy = DDIMStrategy(num_timesteps=1000, beta_schedule='sigmoid', eta=0.0)
        
        # Test forward process
        x0 = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        noise = torch.randn_like(x0)
        
        xt = strategy.forward_process(x0, t, noise)
        
        assert xt.shape == x0.shape, "Output shape mismatch"
        assert not torch.isnan(xt).any(), "NaN in output"
        
        print(f"  Strategy created with sigmoid schedule")
        print(f"  Forward process output shape: {xt.shape}")
        print("✓ DDIM with sigmoid schedule works correctly")
        return True
    except Exception as e:
        print(f"✗ DDIM with sigmoid schedule failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_eta_parameter():
    """Test DDIM eta parameter for stochasticity control"""
    print("\nTesting DDIM eta parameter...")
    try:
        from nnunetv2.training.diffusion import DDIMStrategy
        
        # Test deterministic (eta=0)
        strategy_det = DDIMStrategy(num_timesteps=1000, eta=0.0)
        
        # Test stochastic (eta=1)
        strategy_stoch = DDIMStrategy(num_timesteps=1000, eta=1.0)
        
        assert strategy_det.eta == 0.0, "Eta not set correctly"
        assert strategy_stoch.eta == 1.0, "Eta not set correctly"
        
        print(f"  Deterministic strategy (eta=0.0): ✓")
        print(f"  Stochastic strategy (eta=1.0): ✓")
        print("✓ DDIM eta parameter works correctly")
        return True
    except Exception as e:
        print(f"✗ DDIM eta parameter failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_schedule_comparison():
    """Compare all three noise schedules"""
    print("\nTesting schedule comparison...")
    try:
        from nnunetv2.training.diffusion.noise_schedulers import (
            linear_beta_schedule,
            cosine_beta_schedule,
            sigmoid_beta_schedule
        )
        
        num_timesteps = 1000
        
        linear = linear_beta_schedule(num_timesteps)
        cosine = cosine_beta_schedule(num_timesteps)
        sigmoid = sigmoid_beta_schedule(num_timesteps)
        
        print(f"  Linear:  [{linear.min():.6f}, {linear.max():.6f}]")
        print(f"  Cosine:  [{cosine.min():.6f}, {cosine.max():.6f}]")
        print(f"  Sigmoid: [{sigmoid.min():.6f}, {sigmoid.max():.6f}]")
        
        # All should be in valid range
        for name, schedule in [('Linear', linear), ('Cosine', cosine), ('Sigmoid', sigmoid)]:
            assert (schedule >= 0).all() and (schedule <= 1).all(), f"{name} out of range"
        
        print("✓ All schedules produce valid beta values")
        return True
    except Exception as e:
        print(f"✗ Schedule comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 2 validation tests"""
    print("=" * 70)
    print("Phase 2 Validation: nn-diffusion Framework")
    print("=" * 70)
    
    results = []
    
    # Run all tests
    results.append(("Imports", test_imports()))
    results.append(("DDIM Forward Process", test_ddim_forward_process()))
    results.append(("DDIM Sampling Step", test_ddim_sample_step()))
    results.append(("DDIM Skip Sampling", test_ddim_skip_sampling()))
    results.append(("Sigmoid Schedule", test_sigmoid_schedule()))
    results.append(("DDIM with Sigmoid", test_ddim_with_sigmoid()))
    results.append(("DDIM Eta Parameter", test_eta_parameter()))
    results.append(("Schedule Comparison", test_schedule_comparison()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All Phase 2 validation tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
