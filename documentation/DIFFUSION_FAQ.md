# Diffusion Framework - Frequently Asked Questions

## General Questions

### Q: What's the difference between deterministic and diffusion methods?

**Deterministic methods (L1/MAE, AFP)**:
- Fast training and inference
- Deterministic output (same input → same output)
- Good for real-time applications
- May produce blurry results

**Diffusion methods (DDPM, etc.)**:
- Slower training and inference
- Stochastic output (can generate diverse results)
- Higher quality, more detailed images
- Better for offline, high-quality generation

### Q: Which method should I use?

- **Fast inference needed**: Use L1/MAE trainer
- **Best quality**: Use DDPM with cosine schedule
- **Perceptual quality**: Use AFP (deterministic) or diffusion
- **Multiple outputs**: Use diffusion (can sample multiple times)

### Q: Can I convert between deterministic and diffusion models?

Partially - you can initialize a diffusion model from a deterministic checkpoint, but the reverse requires retraining.

## Technical Questions

### Q: Why does diffusion training take so long?

Diffusion models process each sample multiple times (once per timestep during sampling), and training requires learning across all timesteps. This is inherent to the method.

### Q: How many timesteps should I use?

- **Fast experiments**: 100-500 timesteps
- **Production**: 1000 timesteps (standard)
- **High quality**: 1000-2000 timesteps

### Q: What's the difference between linear and cosine schedules?

- **Linear**: Simple, works well for lower resolutions
- **Cosine**: Smoother noise addition, better for high-resolution images
- Cosine tends to preserve more detail in early training stages

### Q: Can I use this on 2D images?

Yes! The framework supports both 2D and 3D. Use `2d` configuration instead of `3d_fullres`.

## Performance Questions

### Q: How can I speed up training?

1. Reduce `num_timesteps` (e.g., 500 instead of 1000)
2. Use fewer epochs
3. Use multi-GPU training
4. Reduce patch size (via custom plans)

### Q: How can I speed up inference?

1. Wait for DDIM implementation (Phase 2) - 10-50x faster
2. Use deterministic methods for real-time needs
3. Use fewer sampling steps (with DDIM)

### Q: Out of memory - what to do?

1. Reduce `num_timesteps`
2. Reduce batch size (smaller patch size)
3. Use gradient checkpointing (future feature)
4. Use mixed precision (already enabled)

## Troubleshooting

### Q: Training loss not decreasing

1. Check your data preprocessing
2. Try cosine schedule instead of linear
3. Reduce learning rate
4. Verify conditioning is working correctly

### Q: Generated images are noisy

1. Increase number of sampling steps
2. Check if model trained long enough
3. Verify correct checkpoint is loaded

### Q: Colors/intensities are off

1. Check data normalization in preprocessing
2. Verify dataset.json is correct
3. May need to adjust intensity rescaling

## Future Features

### Q: When will DDIM be available?

Phase 2 (estimated: next release)

### Q: When will Flow Matching be available?

Phase 2 (estimated: next release)

### Q: Will you support latent diffusion?

Planned for Phase 4

### Q: Can I use this for unconditional generation?

Not currently - this is designed for conditioned image-to-image translation. Unconditional generation may be added in future phases.

## Data Questions

### Q: Do I need paired data?

Yes, currently the framework requires paired source-target images for training.

### Q: Can I mix different datasets?

Yes, follow nnUNet's multi-dataset training procedures.

### Q: What image formats are supported?

Same as nnUNet: NIfTI (.nii.gz), plus others via preprocessing.

## Citation

### Q: How should I cite this work?

See the main README.md for citation information for both the diffusion framework and original nnUNet_translation.
