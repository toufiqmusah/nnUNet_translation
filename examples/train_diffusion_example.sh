#!/bin/bash

# Example training script for nn-diffusion framework
# This demonstrates various training configurations

# Set paths (update these for your system)
export nnUNet_raw="/data/nnUNet/raw"
export nnUNet_preprocessed="/data/nnUNet/preprocessed"
export nnUNet_results="/data/nnUNet/results"

DATASET=101
CONFIG="3d_fullres"
FOLD=0
PLANS="nnResUNetPlans"

echo "=== Example 1: Basic DDPM Training ==="
nnUNetv2_train $DATASET $CONFIG $FOLD \
  -tr nnUNetDiffusionTrainer \
  -pl $PLANS

echo "=== Example 2: DDPM with Cosine Schedule ==="
nnUNetv2_train $DATASET $CONFIG $FOLD \
  -tr nnUNetDiffusionTrainer \
  -pl $PLANS \
  --beta_schedule cosine

echo "=== Example 3: Fast Training (500 timesteps) ==="
nnUNetv2_train $DATASET $CONFIG $FOLD \
  -tr nnUNetDiffusionTrainer \
  -pl $PLANS \
  --num_timesteps 500

echo "=== Example 4: High Quality (2000 timesteps + cosine) ==="
nnUNetv2_train $DATASET $CONFIG $FOLD \
  -tr nnUNetDiffusionTrainer \
  -pl $PLANS \
  --num_timesteps 2000 \
  --beta_schedule cosine

echo "Training complete! Check results in: $nnUNet_results"
