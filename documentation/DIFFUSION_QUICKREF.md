# nn-diffusion Quick Reference

## 🚀 Quick Commands

### Training
```bash
# Basic DDPM
nnUNetv2_train <dataset> 3d_fullres <fold> -tr nnUNetDiffusionTrainer -pl nnResUNetPlans

# With cosine schedule
--beta_schedule cosine

# Custom timesteps
--num_timesteps 500
```

### Inference
```bash
# Standard
nnUNetv2_predict -d <dataset> -i INPUT -o OUTPUT -c 3d_fullres -p nnResUNetPlans -tr nnUNetDiffusionTrainer -f <fold>
```

## 📊 Strategy Comparison

| Strategy | Status | Training Speed | Quality | Best For |
|----------|--------|----------------|---------|----------|
| DDPM | ✅ Ready | Slow | High | High-quality generation |
| DDIM | 🔄 Phase 2 | Medium | High | Faster inference |
| Flow Matching | 🔄 Phase 2 | Fast | Very High | Efficient training |
| Brownian Bridge | 🔄 Phase 2 | Medium | Very High | Paired translation |

## 🎛️ Key Parameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `num_timesteps` | 1000 | 100-2000 | Quality vs Speed |
| `beta_schedule` | linear | linear/cosine | Noise distribution |
| `beta_start` | 1e-4 | 1e-5 to 1e-3 | Initial noise |
| `beta_end` | 0.02 | 0.01-0.05 | Final noise |

## 💡 Tips

- **High quality**: Use `cosine` schedule with 1000+ timesteps
- **Fast training**: Use 500 timesteps with `linear` schedule
- **Debugging**: Start with 100 timesteps to verify everything works
- **Multi-GPU**: Works with standard DDP (same as nnUNet)

## 🐛 Common Issues

**OOM**: Reduce `num_timesteps` or batch size
**Unstable**: Use `cosine` schedule, reduce learning rate
**Slow inference**: Wait for DDIM (Phase 2) or use deterministic methods
