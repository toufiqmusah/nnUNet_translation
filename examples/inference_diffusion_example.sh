#!/bin/bash

# Example inference script for diffusion models

export nnUNet_results="/data/nnUNet/results"

DATASET=101
CONFIG="3d_fullres"
FOLD=0
PLANS="nnResUNetPlans"
INPUT_DIR="/path/to/input"
OUTPUT_DIR="/path/to/output"

echo "=== Running Diffusion Inference ==="
nnUNetv2_predict -d $DATASET \
  -i $INPUT_DIR \
  -o $OUTPUT_DIR \
  -c $CONFIG \
  -p $PLANS \
  -tr nnUNetDiffusionTrainer \
  -f $FOLD

echo "Results saved to: $OUTPUT_DIR"
