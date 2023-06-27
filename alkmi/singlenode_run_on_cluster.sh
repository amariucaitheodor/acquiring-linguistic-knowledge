#!/bin/bash

NUM_GPUS=${2-1}
VRAM_PER_GPU=${3-20g}
GPU_TYPE=${4-} # a100-pcie-40gb
echo "Selected configuration: $1, GPUs: $NUM_GPUS, $VRAM_PER_GPU VRAM/GPU (type selected: $GPU_TYPE)"

# 5 days is the upper limit on Euler (before PartitionTimeLimit kicks in)

# N.B. If you set the job name to `bash` or `interactive`, Lightning’s SLURM auto-detection
# will get bypassed and it can launch processes normally. This is apparently needed for single node multi-GPU runs...
sbatch --job-name="bash" \
  --time=5-00:00:00 \
  --nodes=1 \
  --ntasks-per-node="$NUM_GPUS" \
  --gpus="$GPU_TYPE":"$NUM_GPUS" \
  --cpus-per-task=4 \
  --mem-per-cpu=15000 \
  --gres=gpumem:"$VRAM_PER_GPU" \
  --output "singlenode_run_$(date "+%F-%T").log" \
  --wrap="WANDB_RUN_GROUP=DDP-$(date "+%F-%T") WANDB__SERVICE_WAIT=300 NUMEXPR_MAX_THREADS=64 python -m train config=$1"
