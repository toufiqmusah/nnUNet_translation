# nn-diffusion Framework - Complete Usage Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Basic Concepts](#basic-concepts)
4. [Training](#training)
5. [Inference](#inference)
6. [Configuration Options](#configuration-options)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)
9. [Examples](#examples)

## Introduction

The nn-diffusion framework extends nnUNet_translation with state-of-the-art diffusion models for medical image-to-image translation. This enables high-quality, diverse image generation with swappable training strategies.

### What are Diffusion Models?

Diffusion models learn to gradually denoise images, starting from pure noise and conditioning on source images (e.g., MR) to generate target images (e.g., CT). This approach has shown superior quality compared to deterministic methods.

### Architecture Overview

```
Source Image (MR) → [Conditioning] → Diffusion Model → Target Image (CT)
                                            ↑
                                     Time Embedding
```

The framework consists of:
- **Diffusion Strategies**: Different training/sampling methods (DDPM, DDIM, etc.)
- **Noise Schedulers**: Control how noise is added/removed (linear, cosine)
- **Conditioning Methods**: How source images guide generation (concatenation, cross-attention)

## Installation

```bash
# Clone the repository
git clone https://github.com/toufiqmusah/nnUNet_translation
cd nnUNet_translation

# Install with diffusion dependencies
pip install -e .
```

The diffusion framework is included by default - no additional installation needed!

## Basic Concepts

### Timesteps

Diffusion models operate over multiple timesteps (typically 1000):
- Training: Random timestep sampled per batch
- Inference: Iterate through all timesteps (or subset with DDIM)

### Noise Schedules

Control the amount of noise at each timestep:

- **Linear**: Simple linear increase (default, works well for most cases)
- **Cosine**: Smoother schedule (better for high-resolution images)
- **Sigmoid**: S-curve schedule (experimental)

### Conditioning

How the source image guides generation:

- **Concatenation**: Source and noisy target concatenated channel-wise (Phase 1)
- **Cross-Attention**: Source features attend to target (Phase 3)

## Training

### Basic Training

```bash
# Set environment variables (same as standard nnUNet)
export nnUNet_raw="/data/nnUNet/raw"
export nnUNet_preprocessed="/data/nnUNet/preprocessed"
export nnUNet_results="/data/nnUNet/results"

# Train DDPM model
nnUNetv2_train 101 3d_fullres 0 \
  -tr nnUNetDiffusionTrainer \
  -pl nnResUNetPlans
```

### Training with Different Noise Schedules

```bash
# Linear schedule (default)
nnUNetv2_train 101 3d_fullres 0 \
  -tr nnUNetDiffusionTrainer \
  -pl nnResUNetPlans \
  --beta_schedule linear

# Cosine schedule (recommended for high-res)
nnUNetv2_train 101 3d_fullres 0 \
  -tr nnUNetDiffusionTrainer \
  -pl nnResUNetPlans \
  --beta_schedule cosine
```

### Adjusting Timesteps

```bash
# Fewer timesteps (faster training, slightly lower quality)
nnUNetv2_train 101 3d_fullres 0 \
  -tr nnUNetDiffusionTrainer \
  -pl nnResUNetPlans \
  --num_timesteps 500

# More timesteps (slower, potentially higher quality)
nnUNetv2_train 101 3d_fullres 0 \
  -tr nnUNetDiffusionTrainer \
  -pl nnResUNetPlans \
  --num_timesteps 2000
```

### Multi-GPU Training

```bash
# DDP training (same as standard nnUNet)
CUDA_VISIBLE_DEVICES=0,1,2,3 nnUNetv2_train 101 3d_fullres 0 \
  -tr nnUNetDiffusionTrainer \
  -pl nnResUNetPlans
```

### Resume Training

```bash
# Resume from checkpoint
nnUNetv2_train 101 3d_fullres 0 \
  -tr nnUNetDiffusionTrainer \
  -pl nnResUNetPlans \
  --continue_training
```

## Inference

### Basic Inference

```bash
# Generate translations
nnUNetv2_predict -d 101 \
  -i /path/to/input \
  -o /path/to/output \
  -c 3d_fullres \
  -p nnResUNetPlans \
  -tr nnUNetDiffusionTrainer \
  -f 0
```

### Fast Inference with Fewer Steps

For DDIM sampling (when implemented in Phase 2), you can use fewer steps:

```bash
# Fast inference (50 steps instead of 1000)
nnUNetv2_predict -d 101 \
  -i /path/to/input \
  -o /path/to/output \
  -c 3d_fullres \
  -p nnResUNetPlans \
  -tr nnUNetDiffusionTrainer \
  -f 0 \
  --sampling_steps 50
```

### Generating Multiple Diverse Outputs

```bash
# Generate 5 diverse translations per input
nnUNetv2_predict -d 101 \
  -i /path/to/input \
  -o /path/to/output \
  -c 3d_fullres \
  -p nnResUNetPlans \
  -tr nnUNetDiffusionTrainer \
  -f 0 \
  --num_samples 5
```

## Configuration Options

### Trainer Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--diffusion_strategy` | `ddpm` | Strategy to use (ddpm, ddim, flow_matching, brownian_bridge) |
| `--num_timesteps` | `1000` | Number of diffusion timesteps |
| `--beta_schedule` | `linear` | Noise schedule (linear, cosine, sigmoid) |
| `--beta_start` | `1e-4` | Starting beta value |
| `--beta_end` | `0.02` | Ending beta value |

### Performance Tuning

```bash
# For faster training (smaller batch, fewer iterations)
nnUNetv2_train 101 3d_fullres 0 \
  -tr nnUNetDiffusionTrainer \
  -pl nnResUNetPlans \
  --num_epochs 300 \
  --num_timesteps 500

# For best quality (more epochs, more timesteps)
nnUNetv2_train 101 3d_fullres 0 \
  -tr nnUNetDiffusionTrainer \
  -pl nnResUNetPlans \
  --num_epochs 1000 \
  --num_timesteps 2000 \
  --beta_schedule cosine
```

## Advanced Usage

### Custom Diffusion Strategy

You can implement your own diffusion strategy by extending `DiffusionStrategy`:

```python
from nnunetv2.training.diffusion import DiffusionStrategy

class MyCustomStrategy(DiffusionStrategy):
    def forward_process(self, x0, t, noise, **kwargs):
        # Your forward diffusion logic
        pass
    
    def compute_loss(self, model_output, target, t, **kwargs):
        # Your loss computation
        pass
    
    def sample_step(self, xt, t, model, conditioning=None, **kwargs):
        # Your sampling logic
        pass
    
    def get_target(self, x0, noise, t, **kwargs):
        # What the model should predict
        pass
```

### Monitoring Training

Training logs are saved in the standard nnUNet results folder:

```bash
# Monitor training progress
tensorboard --logdir $nnUNet_results/Dataset101_YourDataset/nnUNetDiffusionTrainer__nnResUNetPlans__3d_fullres/fold_0
```

### Comparing Methods

```bash
# Train deterministic baseline
nnUNetv2_train 101 3d_fullres 0 -tr nnUNetTrainerMRCT_mae -pl nnResUNetPlans

# Train diffusion model
nnUNetv2_train 101 3d_fullres 0 -tr nnUNetDiffusionTrainer -pl nnResUNetPlans

# Compare results using your preferred metrics (PSNR, SSIM, etc.)
```

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce number of timesteps
--num_timesteps 500

# Or reduce batch size (modify plans or use smaller patch size)
nnUNetv2_plan_experiment -d 101 -c 3d_fullres -pl nnUNetPlannerResUNet --patch_size 64 64 64
```

### Training Instability

```bash
# Use cosine schedule (more stable)
--beta_schedule cosine

# Reduce learning rate
--initial_lr 5e-5
```

### Slow Inference

```bash
# Use DDIM with fewer steps (Phase 2)
--sampling_steps 50

# Or use deterministic methods for real-time applications
-tr nnUNetTrainerMRCT_mae
```

## Examples

### Example 1: MR to CT Translation

```bash
# 1. Prepare dataset (same as nnUNet format)
# Dataset101_MRCT/
#   ├── imagesTr/
#   │   ├── case_001_0000.nii.gz  # MR image
#   │   └── ...
#   ├── labelsTr/
#   │   ├── case_001.nii.gz       # CT image (as "label")
#   │   └── ...
#   └── dataset.json

# 2. Preprocess
nnUNetv2_plan_and_preprocess -d 101 -c 3d_fullres

# 3. Train diffusion model
nnUNetv2_train 101 3d_fullres 0 \
  -tr nnUNetDiffusionTrainer \
  -pl nnResUNetPlans \
  --beta_schedule cosine

# 4. Inference
nnUNetv2_predict -d 101 -i INPUT -o OUTPUT -c 3d_fullres \
  -p nnResUNetPlans -tr nnUNetDiffusionTrainer -f 0
```

### Example 2: Fine-tuning from Deterministic Model

```bash
# 1. Train deterministic baseline
nnUNetv2_train 101 3d_fullres 0 -tr nnUNetTrainerMRCT_mae -pl nnResUNetPlans

# 2. Fine-tune with diffusion
nnUNetv2_train 101 3d_fullres 0 \
  -tr nnUNetDiffusionTrainer \
  -pl nnResUNetPlans \
  -pretrained_weights PATH_TO_BASELINE_CHECKPOINT
```

### Example 3: Comparing Noise Schedules

```bash
# Linear schedule
nnUNetv2_train 101 3d_fullres 0 -tr nnUNetDiffusionTrainer \
  -pl nnResUNetPlans --beta_schedule linear

# Cosine schedule  
nnUNetv2_train 101 3d_fullres 1 -tr nnUNetDiffusionTrainer \
  -pl nnResUNetPlans --beta_schedule cosine

# Compare validation losses and image quality
```

## Next Steps

- **Phase 2**: DDIM, Flow Matching, Brownian Bridge implementations
- **Phase 3**: Cross-attention conditioning, ControlNet-style guidance
- **Phase 4**: Latent diffusion models, multi-scale architectures

## References

- Ho et al. (2020) - "Denoising Diffusion Probabilistic Models"
- Song et al. (2020) - "Denoising Diffusion Implicit Models"
- Rombach et al. (2022) - "High-Resolution Image Synthesis with Latent Diffusion Models"
- Longuefosse et al. (2024) - "Adapted nnU-Net: A Robust Baseline for Cross-Modality Synthesis"

## Citation

If you use the diffusion framework, please cite:

```bibtex
@inproceedings{longuefosse2024adapted,
  title={Adapted nnU-Net: A Robust Baseline for Cross-Modality Synthesis and Medical Image Inpainting},
  author={Longuefosse, Arthur and Bot, Emma Le and De Senneville, Baudouin Denis and Giraud, Romain and Mansencal, Boris and Coup{\'e}, Pierrick and Baldacci, Fanny},
  booktitle={International Workshop on Simulation and Synthesis in Medical Imaging},
  pages={24--33},
  year={2024},
  organization={Springer}
}
```
