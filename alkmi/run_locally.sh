#!/bin/bash

echo "Selected configuration for a local run: $1"

CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 WANDB__SERVICE_WAIT=300 NUMEXPR_MAX_THREADS=64 python -m train config=$1
