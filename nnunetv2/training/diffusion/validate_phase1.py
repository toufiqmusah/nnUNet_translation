#!/usr/bin/env python3
"""
Validation script for Phase 1 of the nn-diffusion framework.
Tests basic functionality of implemented components.
"""

import sys
import torch
import numpy as np


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from nnunetv2.training.diffusion import (
            DiffusionStrategy,
            DDPMStrategy
        )
        from nnunetv2.training.diffusion.conditioning import (
            ConditioningMethod,
            ConcatConditioning
        )
        from nnunetv2.training.diffusion.noise_schedulers import (
            linear_beta_schedule,
            cosine_beta_schedule
        )
        from nnunetv2.training.diffusion.utils import (
            SinusoidalTimeEmbedding,
            TimeEmbeddingMLP,
            extract_into_tensor,
            normalize_to_neg_one_to_one,
            unnormalize_to_zero_to_one
        )
        print("✓ All core imports successful")
        print("  Note: nnUNetDiffusionTrainer requires full nnUNet dependencies")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ddpm_forward_process():
    """Test DDPM forward diffusion process"""
    print("\nTesting DDPM forward process...")
    try:
        from nnunetv2.training.diffusion import DDPMStrategy
        
        # Create strategy
        strategy = DDPMStrategy(num_timesteps=1000, beta_schedule='linear')
        
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
        print("✓ DDPM forward process works correctly")
        return True
    except Exception as e:
        print(f"✗ DDPM forward process failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ddpm_loss():
    """Test DDPM loss computation"""
    print("\nTesting DDPM loss computation...")
    try:
        from nnunetv2.training.diffusion import DDPMStrategy
        
        strategy = DDPMStrategy(num_timesteps=1000)
        
        # Create test data
        model_output = torch.randn(2, 3, 32, 32)
        target = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        
        # Compute loss
        loss = strategy.compute_loss(model_output, target, t)
        
        assert loss.ndim == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss is NaN"
        
        print(f"  Loss value: {loss.item():.6f}")
        print("✓ DDPM loss computation works correctly")
        return True
    except Exception as e:
        print(f"✗ DDPM loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_time_embeddings():
    """Test sinusoidal time embeddings"""
    print("\nTesting time embeddings...")
    try:
        from nnunetv2.training.diffusion.utils import SinusoidalTimeEmbedding, TimeEmbeddingMLP
        
        # Test SinusoidalTimeEmbedding
        embed_dim = 256
        embedder = SinusoidalTimeEmbedding(embed_dim)
        
        t = torch.tensor([0, 100, 500, 999])
        emb = embedder(t)
        
        assert emb.shape == (4, embed_dim), f"Expected shape (4, {embed_dim}), got {emb.shape}"
        assert not torch.isnan(emb).any(), "NaN in embeddings"
        
        print(f"  Embedding shape: {emb.shape}")
        print(f"  Embedding range: [{emb.min():.3f}, {emb.max():.3f}]")
        
        # Test TimeEmbeddingMLP
        mlp = TimeEmbeddingMLP(embed_dim)
        emb_processed = mlp(emb)
        
        assert emb_processed.shape == emb.shape, "MLP changed embedding shape"
        assert not torch.isnan(emb_processed).any(), "NaN after MLP"
        
        print(f"  MLP output shape: {emb_processed.shape}")
        print("✓ Time embeddings work correctly")
        return True
    except Exception as e:
        print(f"✗ Time embeddings failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_noise_schedulers():
    """Test noise schedulers"""
    print("\nTesting noise schedulers...")
    try:
        from nnunetv2.training.diffusion.noise_schedulers import (
            linear_beta_schedule,
            cosine_beta_schedule
        )
        
        num_timesteps = 1000
        
        # Test linear schedule
        linear_betas = linear_beta_schedule(num_timesteps)
        assert linear_betas.shape == (num_timesteps,), "Linear schedule shape mismatch"
        assert (linear_betas >= 0).all() and (linear_betas <= 1).all(), "Betas out of range"
        print(f"  Linear schedule: {linear_betas[:3].tolist()} ... {linear_betas[-3:].tolist()}")
        
        # Test cosine schedule
        cosine_betas = cosine_beta_schedule(num_timesteps)
        assert cosine_betas.shape == (num_timesteps,), "Cosine schedule shape mismatch"
        assert (cosine_betas >= 0).all() and (cosine_betas <= 1).all(), "Betas out of range"
        print(f"  Cosine schedule: {cosine_betas[:3].tolist()} ... {cosine_betas[-3:].tolist()}")
        
        print("✓ Noise schedulers work correctly")
        return True
    except Exception as e:
        print(f"✗ Noise schedulers failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_conditioning():
    """Test conditioning methods"""
    print("\nTesting conditioning methods...")
    try:
        from nnunetv2.training.diffusion.conditioning import ConcatConditioning
        
        conditioning = ConcatConditioning()
        
        # Test data
        x = torch.randn(2, 3, 32, 32)
        source = torch.randn(2, 3, 32, 32)
        
        # Prepare and apply conditioning
        cond = conditioning.prepare_conditioning(source)
        output = conditioning.apply_conditioning(x, cond)
        
        assert output.shape == (2, 6, 32, 32), f"Expected shape (2, 6, 32, 32), got {output.shape}"
        print(f"  Input shape: {x.shape}")
        print(f"  Conditioning shape: {cond.shape}")
        print(f"  Output shape: {output.shape}")
        print("✓ Conditioning methods work correctly")
        return True
    except Exception as e:
        print(f"✗ Conditioning methods failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_helper_functions():
    """Test utility helper functions"""
    print("\nTesting helper functions...")
    try:
        from nnunetv2.training.diffusion.utils import (
            normalize_to_neg_one_to_one,
            unnormalize_to_zero_to_one
        )
        
        # Test normalization
        img = torch.rand(2, 3, 32, 32)  # [0, 1]
        normalized = normalize_to_neg_one_to_one(img)
        assert normalized.min() >= -1.0 and normalized.max() <= 1.0, "Normalization out of range"
        
        # Test unnormalization
        unnormalized = unnormalize_to_zero_to_one(normalized)
        assert torch.allclose(img, unnormalized, atol=1e-6), "Round-trip normalization failed"
        
        print(f"  Original range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"  Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
        print(f"  Unnormalized range: [{unnormalized.min():.3f}, {unnormalized.max():.3f}]")
        print("✓ Helper functions work correctly")
        return True
    except Exception as e:
        print(f"✗ Helper functions failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests"""
    print("=" * 70)
    print("Phase 1 Validation: nn-diffusion Framework")
    print("=" * 70)
    
    results = []
    
    # Run all tests
    results.append(("Imports", test_imports()))
    results.append(("DDPM Forward Process", test_ddpm_forward_process()))
    results.append(("DDPM Loss", test_ddpm_loss()))
    results.append(("Time Embeddings", test_time_embeddings()))
    results.append(("Noise Schedulers", test_noise_schedulers()))
    results.append(("Conditioning", test_conditioning()))
    results.append(("Helper Functions", test_helper_functions()))
    
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
        print("\n🎉 All validation tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
