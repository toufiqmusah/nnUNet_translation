# nn-diffusion: Modular Diffusion Framework for Medical Image Translation

## Overview

The nn-diffusion framework provides a plug-and-play diffusion-based architecture for image-to-image translation tasks in medical imaging. Built on top of nnUNet, it enables researchers to easily experiment with different diffusion strategies, noise schedules, and conditioning methods.

## Architecture

The framework is organized into modular components:

```
nnunetv2/training/diffusion/
├── base_diffusion_strategy.py      # Abstract base for diffusion strategies
├── base_diffusion_trainer.py       # Base trainer extending nnUNetTrainer
├── schedulers/                      # Different diffusion strategies
│   ├── ddpm.py                      # ✓ DDPM implementation
│   ├── ddim.py                      # ✓ DDIM implementation (Phase 2)
│   ├── flow_matching.py             # TODO: Phase 3
│   └── brownian_bridge.py           # TODO: Phase 3
├── noise_schedulers/                # Beta/noise schedules
│   ├── linear.py                    # ✓ Linear schedule
│   ├── cosine.py                    # ✓ Cosine schedule
│   └── sigmoid.py                   # ✓ Sigmoid schedule (Phase 2)
├── conditioning/                    # Conditioning methods
│   ├── base_conditioning.py         # Abstract base
│   ├── concat_conditioning.py       # ✓ Channel concatenation
│   └── cross_attention_conditioning.py  # TODO: Future
└── utils/                           # Utility functions
    ├── time_embedding.py            # ✓ Sinusoidal embeddings
    ├── helpers.py                   # ✓ Helper functions
    └── sampling.py                  # ✓ Sampling utilities (Phase 2)
```

## Key Components

### 1. Diffusion Strategies

The `DiffusionStrategy` abstract base class defines the interface for all diffusion methods:

- `forward_process()`: Adds noise to clean data
- `get_target()`: Computes training target (e.g., noise for DDPM)
- `compute_loss()`: Calculates training loss
- `sample_step()`: Performs one denoising step during inference

**Currently Implemented:**
- **DDPM** (Denoising Diffusion Probabilistic Models): The foundational diffusion strategy with noise prediction
- **DDIM** (Denoising Diffusion Implicit Models): Deterministic sampling for 10-20x faster inference ✨ NEW

**Coming Soon:**
- Flow Matching: Optimal transport-based training
- Brownian Bridge: Direct source-to-target diffusion

### 2. Noise Schedulers

Control how noise is added over time:

- **Linear**: Simple linear increase (Ho et al., 2020)
- **Cosine**: Better for high-resolution images (Nichol & Dhariwal, 2021)
- **Sigmoid**: Smoother transitions with S-curve ✨ NEW

### 3. Conditioning Methods

How source information is provided to the model:

- **Concat Conditioning**: Simple channel-wise concatenation

### 4. Time Embeddings

- **SinusoidalTimeEmbedding**: Positional encoding for timesteps
- **TimeEmbeddingMLP**: MLP to process time information

### 5. Sampling Utilities ✨ NEW

- **ddpm_sample_loop**: Full sampling loop for DDPM
- **ddim_sample_loop**: Fast sampling loop for DDIM (with timestep skipping)
- **sample_with_strategy**: Unified interface for both strategies
- **progressive_sampling**: Save intermediate denoising steps
- **interpolate_samples**: Interpolate between samples in latent space

## Usage

### Basic Training Example

```python
from nnunetv2.training.diffusion import nnUNetDiffusionTrainer

# Create trainer with DDPM strategy
trainer = nnUNetDiffusionTrainer(
    plans=plans,
    configuration=configuration,
    fold=fold,
    dataset_json=dataset_json,
    diffusion_strategy='ddpm',
    num_timesteps=1000,
    beta_schedule='linear'
)

# Initialize and train
trainer.initialize()
trainer.run_training()
```

### Using DDIM for Faster Inference ✨ NEW

```python
from nnunetv2.training.diffusion import nnUNetDiffusionTrainer

# Train with DDPM (or DDIM - same training process)
trainer = nnUNetDiffusionTrainer(
    plans=plans,
    configuration=configuration,
    fold=fold,
    dataset_json=dataset_json,
    diffusion_strategy='ddpm',
    num_timesteps=1000,
    beta_schedule='cosine'
)
trainer.initialize()
trainer.run_training()

# Inference with DDIM (10-20x faster!)
trainer_ddim = nnUNetDiffusionTrainer(
    plans=plans,
    configuration=configuration,
    fold=fold,
    dataset_json=dataset_json,
    diffusion_strategy='ddim',  # Use DDIM for inference
    num_timesteps=1000,
    beta_schedule='cosine'
)
trainer_ddim.initialize()
# Load trained weights...
# Run fast inference with only 50 steps instead of 1000!
```

### Using Different Noise Schedules

```python
# Linear schedule (simple, stable)
trainer_linear = nnUNetDiffusionTrainer(..., beta_schedule='linear')

# Cosine schedule (better for high-resolution)
trainer_cosine = nnUNetDiffusionTrainer(..., beta_schedule='cosine')

# Sigmoid schedule (smooth transitions) ✨ NEW
trainer_sigmoid = nnUNetDiffusionTrainer(..., beta_schedule='sigmoid')
```

### Advanced Sampling ✨ NEW

```python
from nnunetv2.training.diffusion.utils import ddim_sample_loop, sample_with_strategy

# Fast DDIM sampling with only 50 steps
samples = ddim_sample_loop(
    model=trainer.network,
    diffusion_strategy=trainer.diffusion_strategy,
    shape=(4, 3, 256, 256),
    conditioning=source_images,
    num_inference_steps=50,  # Only 50 steps instead of 1000!
    progress=True
)

# Or use unified interface
samples = sample_with_strategy(
    model=trainer.network,
    diffusion_strategy=trainer.diffusion_strategy,
    shape=(4, 3, 256, 256),
    strategy_name='ddim',  # or 'ddpm'
    conditioning=source_images,
    num_inference_steps=50
)
```
    configuration=configuration,
    fold=fold,
    dataset_json=dataset_json,
    diffusion_strategy='ddpm',
    num_timesteps=1000,
    beta_schedule='cosine'  # Changed from 'linear'
)
```

## Key Features

### Modularity
- Easy to swap different diffusion strategies
- Plug-and-play noise schedules
- Flexible conditioning methods

### Extensibility
- Abstract base classes make it easy to add new strategies
- All components follow consistent interfaces
- Well-documented code with type hints

### Integration
- Built on top of proven nnUNet architecture
- Maintains compatibility with existing nnUNet features
- Leverages nnUNet's data loading and preprocessing

## Design Philosophy

1. **Simplicity**: Clean, understandable code over clever abstractions
2. **Modularity**: Each component is independent and swappable
3. **Extensibility**: Easy to add new strategies and methods
4. **Backward Compatibility**: Works alongside existing nnUNet functionality

## Implementation Details

### Forward Diffusion Process

The forward process gradually adds noise to clean images:

```
q(x_t | x_0) = N(x_t; √(ᾱ_t) x_0, (1 - ᾱ_t) I)
```

Where:
- `x_0` is the clean image
- `x_t` is the noisy image at timestep t
- `ᾱ_t` is the cumulative product of alphas

### Training Objective

DDPM trains a neural network to predict the noise:

```
L = E[||ε - ε_θ(x_t, t)||²]
```

Where:
- `ε` is the true noise
- `ε_θ(x_t, t)` is the predicted noise from the model

### Reverse Process

During sampling, the model iteratively denoises:

```
x_{t-1} = μ_θ(x_t, t) + σ_t z
```

Where `z ~ N(0, I)` is random noise (except at t=0)

## References

### Phase 1 - Implemented

1. **DDPM**: Ho et al. (2020) - "Denoising Diffusion Probabilistic Models"
   - arXiv: 2006.11239

2. **Cosine Schedule**: Nichol & Dhariwal (2021) - "Improved Denoising Diffusion Probabilistic Models"
   - arXiv: 2102.09672

### Phase 2 - Implemented ✨

3. **DDIM**: Song et al. (2021) - "Denoising Diffusion Implicit Models"
   - arXiv: 2010.02502
   - Enables 10-20x faster inference through deterministic sampling

### Future Work

4. **Flow Matching**: Lipman et al. (2023) - "Flow Matching for Generative Modeling"
   - arXiv: 2210.02747

5. **Brownian Bridge**: Delbracio & Milanfar (2023) - "Inversion by Direct Iteration"
   - arXiv: 2303.11435

## Roadmap

### Phase 1 (Complete) ✅
- [x] Base architecture and abstractions
- [x] DDPM implementation
- [x] Linear and cosine schedules
- [x] Concat conditioning
- [x] Time embeddings
- [x] Base diffusion trainer

### Phase 2 (Complete) ✅
- [x] DDIM for faster sampling (10-20x speedup)
- [x] Sigmoid noise schedule
- [x] Sampling utilities (multiple strategies)
- [x] Progressive and interpolated sampling
- [x] Enhanced trainer with DDIM support

### Phase 3 (Future)
- [ ] Flow matching strategy
- [ ] Brownian bridge for I2I translation
- [ ] Advanced conditioning methods (if needed)
- [ ] Multi-resolution diffusion
- [ ] Classifier-free guidance
- [ ] Network architecture modifications for time embedding injection

## Visualizations

Run the examples script to generate visualizations:

```bash
python nnunetv2/training/diffusion/examples.py
```

This will generate:
- **Noise schedule comparison**: Shows how linear vs cosine schedules differ
- **Time embeddings visualization**: Shows how timesteps are encoded

### Noise Schedules

Different noise schedules affect how quickly noise is added during the forward process:

- **Linear schedule**: Simple linear increase in noise from β_start to β_end
- **Cosine schedule**: More gradual at the beginning, better for high-resolution images

### Time Embeddings

Sinusoidal time embeddings encode timestep information for the model, allowing it to know what timestep it's processing and adjust its denoising accordingly.

## Testing

The framework includes comprehensive validation:

```bash
# Run validation tests
python nnunetv2/training/diffusion/validate_phase1.py

# Run examples
python nnunetv2/training/diffusion/examples.py
```

Example code:

```python
# Test imports
from nnunetv2.training.diffusion import (
    DiffusionStrategy,
    DDPMStrategy
)

# Test DDPM forward process
import torch
strategy = DDPMStrategy(num_timesteps=1000)
x0 = torch.randn(2, 3, 64, 64)
t = torch.randint(0, 1000, (2,))
noise = torch.randn_like(x0)
xt = strategy.forward_process(x0, t, noise)
```

## Contributing

When adding new strategies or components:

1. Inherit from appropriate abstract base class
2. Implement all required methods
3. Add comprehensive docstrings
4. Update this README
5. Add to appropriate `__init__.py`

## Notes

- Phase 1 focuses on foundation - network architecture modifications for time embeddings come in Phase 2
- The trainer currently passes time embeddings but networks need updates to utilize them
- Backward compatible with existing nnUNet_translation functionality

## Support

For issues or questions:
- Check the documentation in each module
- Review the abstract base classes for required interfaces
- See examples in the implemented DDPM strategy

## License

Follows the same license as nnUNet_translation (Apache 2.0)
