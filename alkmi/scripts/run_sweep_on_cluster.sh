#!/bin/bash

NUM_GPUS=${3-4}
VRAM_PER_GPU=${4-40g}
echo "Number of agents: $1, sweep name: $2, GPUs per agent: $NUM_GPUS, $VRAM_PER_GPU VRAM/GPU"

# 5 days is the upper limit on Euler (before PartitionTimeLimit kicks in)

# For more details on running Lightning on SLURM:
# https://github.com/Lightning-AI/lightning/blob/66293758c0192955148e8b4bd9dd156927c23503/src/lightning/fabric/plugins/environments/slurm.py#L195

# N.B. If you set the job name to `bash` or `interactive`, Lightning’s SLURM auto-detection
# will get bypassed and it can launch processes normally. This is apparently needed for single node multi-GPU runs...
for ((i = 1; i <= $1; i++)); do
  NR=$i
  sbatch --job-name="bash" \
    --time=5-00:00:00 \
    --nodes=1 \
    --ntasks="$NUM_GPUS" \
    --ntasks-per-node="$NUM_GPUS" \
    --gpus="$NUM_GPUS" \
    --cpus-per-task=4 \
    --mem-per-cpu=15000 \
    --gres=gpumem:"$VRAM_PER_GPU" \
    --output "sweep_($2)_#${NR}_($(date "+%F-%T")).log" \
    --wrap="WANDB_RUN_GROUP=DDP-$(date "+%F-%T") WANDB__SERVICE_WAIT=300 NUMEXPR_MAX_THREADS=64 wandb agent --count 1 $2"
  sleep 1
done
