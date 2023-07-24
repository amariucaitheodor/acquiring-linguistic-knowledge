#!/bin/bash

NUM_GPUS=${2-4}
VRAM_PER_GPU=${3-40g}
echo "Selected configuration: $1, GPUs: $NUM_GPUS, $VRAM_PER_GPU VRAM/GPU"

# 5 days is the upper limit on Euler (before PartitionTimeLimit kicks in)

# N.B.
# 1. For multiple GPUs, runs will be shown as a group on WandB (not as a single run, which is typically nicer)
# 2. If you set the job name to `bash` or `interactive`, Lightning’s SLURM auto-detection
# will get bypassed and it can launch processes normally. This is apparently needed for single node multi-GPU runs...
sbatch --job-name="bash" \
  --time=5-00:00:00 \
  --nodes=1 \
  --ntasks="$NUM_GPUS" \
  --ntasks-per-node="$NUM_GPUS" \
  --gpus="$NUM_GPUS" \
  --cpus-per-task=4 \
  --mem-per-cpu=10000 \
  --gres=gpumem:"$VRAM_PER_GPU" \
  --output "singlenode_$(date "+%F-%T").log" \
  --wrap="WANDB_RUN_GROUP=DDP-$(date "+%F-%T") WANDB__SERVICE_WAIT=300 NUMEXPR_MAX_THREADS=64 python -m train config=$1"

# eu-a65-03 might have an issue
# --nodelist=eu-a65-02,eu-a65-04,eu-a65-05,eu-a65-07 \