# nn-diffusion Architecture Documentation

## Overview

The nn-diffusion framework is built on a modular, extensible architecture that allows swapping diffusion strategies, noise schedules, and conditioning methods.

## Directory Structure

```
nnunetv2/training/
├── nnUNetTrainer/
│   ├── nnUNetTrainer.py                 # Base trainer
│   ├── nnUNetDiffusionTrainer.py        # Diffusion trainer (NEW LOCATION)
│   └── variants/                        # Other trainer variants
│
└── diffusion/                           # Diffusion framework
    ├── __init__.py                      # Framework exports
    ├── diffusion_strategy.py            # Abstract strategy interface (renamed)
    │
    ├── schedulers/                      # Diffusion strategies
    │   ├── __init__.py
    │   ├── ddpm.py                      # DDPM implementation
    │   ├── ddim.py                      # DDIM (Phase 2)
    │   ├── flow_matching.py             # Flow Matching (Phase 2)
    │   └── brownian_bridge.py           # Brownian Bridge (Phase 2)
    │
    ├── noise_schedulers/                # Beta schedules
    │   ├── __init__.py
    │   ├── linear.py                    # Linear schedule
    │   ├── cosine.py                    # Cosine schedule
    │   └── sigmoid.py                   # Sigmoid schedule (Phase 2)
    │
    ├── conditioning/                    # Conditioning methods
    │   ├── __init__.py
    │   ├── conditioning_base.py         # Abstract interface (renamed)
    │   ├── concat_conditioning.py       # Channel concatenation
    │   └── cross_attention_conditioning.py  # Cross-attention (Phase 3)
    │
    └── utils/                           # Utilities
        ├── __init__.py
        ├── time_embedding.py            # Time embeddings
        └── helpers.py                   # Helper functions
```

**Note**: The `nnUNetDiffusionTrainer` is now located in `nnunetv2/training/nnUNetTrainer/` to follow nnUNet's trainer discovery convention. All trainers must be in this directory to be discoverable by `nnUNetv2_train`.

## Core Components

### 1. DiffusionStrategy (Abstract Base Class)

Defines the interface all strategies must implement:

```python
class DiffusionStrategy(ABC):
    @abstractmethod
    def forward_process(self, x0, t, noise, **kwargs) -> torch.Tensor:
        """Add noise to data"""
        
    @abstractmethod
    def compute_loss(self, model_output, target, t, **kwargs) -> torch.Tensor:
        """Compute training loss"""
        
    @abstractmethod
    def sample_step(self, xt, t, model, conditioning, **kwargs) -> torch.Tensor:
        """One denoising step"""
        
    @abstractmethod
    def get_target(self, x0, noise, t, **kwargs) -> torch.Tensor:
        """What should the model predict?"""
```

### 2. nnUNetDiffusionTrainer

Extends `nnUNetTrainer` with diffusion-specific logic:

- Manages diffusion strategy lifecycle
- Handles time embeddings
- Coordinates conditioning
- Modified training/validation loops

### 3. Conditioning Methods

Control how source images guide generation:

- **ConcatConditioning**: Simple channel-wise concatenation
- **CrossAttentionConditioning**: (Phase 3) Attention-based guidance

### 4. Time Embeddings

- **SinusoidalTimeEmbedding**: Positional encoding for timesteps
- **TimeEmbeddingMLP**: Process embeddings before injection

## Data Flow

### Training

```
1. Load batch: {source_image, target_image}
2. Sample timestep t ~ Uniform(0, T)
3. Sample noise ε ~ N(0, I)
4. Forward diffusion: x_t = forward_process(target, t, ε)
5. Prepare conditioning: c = prepare_conditioning(source)
6. Apply conditioning: x_input = apply_conditioning(x_t, c)
7. Embed time: t_emb = time_embedder(t)
8. Model prediction: pred = network(x_input, t_emb)
9. Get target: target = get_target(target, ε, t)
10. Compute loss: loss = compute_loss(pred, target, t)
11. Backpropagate
```

### Inference

```
1. Load source image
2. Sample noise x_T ~ N(0, I)
3. Prepare conditioning: c = prepare_conditioning(source)
4. For t = T to 1:
   a. Embed time: t_emb = time_embedder(t)
   b. Apply conditioning: x_input = apply_conditioning(x_t, c)
   c. Model prediction: pred = network(x_input, t_emb)
   d. Denoise: x_{t-1} = sample_step(x_t, t, pred, c)
5. Return x_0
```

## Extension Points

### Adding a New Strategy

1. Create new file in `schedulers/`
2. Extend `DiffusionStrategy`
3. Implement required methods
4. Register in `__init__.py`
5. Add to trainer's `_build_diffusion_strategy()`

### Adding a New Noise Schedule

1. Create function in `noise_schedulers/`
2. Return tensor of beta values
3. Use in strategy initialization

### Adding a New Conditioning Method

1. Create file in `conditioning/`
2. Extend `ConditioningMethod`
3. Implement `prepare_conditioning` and `apply_conditioning`
4. Use in trainer initialization

## Design Principles

1. **Modularity**: Each component is independent and swappable
2. **Extensibility**: Easy to add new strategies/methods
3. **Compatibility**: Works with existing nnUNet infrastructure
4. **Flexibility**: Configurable via command-line arguments
5. **Maintainability**: Clear abstractions and documentation

## Future Enhancements

### Phase 2
- DDIM for fast sampling
- Flow Matching for efficient training
- Brownian Bridge for better paired translation

### Phase 3
- Cross-attention conditioning
- ControlNet-style guidance
- Multi-scale architectures

### Phase 4
- Latent diffusion (VAE + diffusion)
- Classifier-free guidance
- Conditional dropout for unconditional generation
