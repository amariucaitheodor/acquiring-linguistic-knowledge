#!/bin/bash

VRAM_PER_GPU=${3-20g}
NUM_GPUS=${4-1}
echo "Number of agents: $1, GPUs per agent: $NUM_GPUS, sweep name: $2, $VRAM_PER_GPU VRAM/GPU"

# 5 days is the upper limit on Euler (before PartitionTimeLimit kicks in)

# N.B. If you set the job name to `bash` or `interactive`, Lightning’s SLURM auto-detection
# will get bypassed and it can launch processes normally. This is apparently needed for single node multi-GPU runs...
for ((i = 1; i <= $1; i++)); do
  NR=$i
  sbatch --job-name="bash" \
    --time=5-00:00:00 \
    --nodes=1 \
    --ntasks-per-node="$NUM_GPUS" \
    --gpus="$NUM_GPUS" \
    --cpus-per-task=4 \
    --mem-per-cpu=15000 \
    --gres=gpumem:"$VRAM_PER_GPU" \
    --output "sweep_($2)_#${NR}_($(date "+%F-%T")).log" \
    --error "sweep_($2)_#${NR}_($(date "+%F-%T")).error" \
    --wrap="WANDB_RUN_GROUP=DDP-$(date "+%F-%T") WANDB__SERVICE_WAIT=300 NUMEXPR_MAX_THREADS=64 wandb agent --count 1 $2"
  sleep 1
done
