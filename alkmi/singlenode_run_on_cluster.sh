#!/bin/bash

NUM_GPUS=${2-1}
VRAM_PER_GPU=${3-20g}
echo "Selected configuration: $1, GPUs: $NUM_GPUS, $VRAM_PER_GPU VRAM/GPU"

# --mail-type=END,FAIL uncomment to get email notifications
sbatch --job-name="multimodal" \
  --time=5-00:00:00 \
  --nodes=1 \
  --ntasks-per-node="$NUM_GPUS" \
  --gpus="$NUM_GPUS" \
  --cpus-per-task=2 \
  --tmp=100G \
  --mem-per-cpu=8000 \
  --gres=gpumem:"$VRAM_PER_GPU" \
  -o "singlenode_run_$(date "+%F-%T").results" \
  --wrap="WANDB__SERVICE_WAIT=300 NUMEXPR_MAX_THREADS=64 python -m train config=$1"
