#!/bin/bash

VRAM_PER_GPU=${3-20g}
NUM_GPUS=${4-1}
echo "Number of agents: $1, GPUs per agent: $NUM_GPUS, sweep name: $2, $VRAM_PER_GPU VRAM/GPU"

# --mail-type=END,FAIL uncomment to get email notifications
# Submit job
for ((i = 1; i <= $1; i++)); do
  NR=$i
  sbatch --job-name="sweep-mm" \
    --time=5-00:00:00 \
    --nodes=1 \
    --ntasks-per-node="$NUM_GPUS" \
    --gpus="$NUM_GPUS" \
    --cpus-per-task=4 \
    --mem-per-cpu=15000 \
    --gres=gpumem:"$VRAM_PER_GPU" \
    -o "sweep_run_($2)_#${NR}_($(date "+%F-%T")).results" \
    --wrap="WANDB_RUN_GROUP=DDP-$(date "+%F-%T") WANDB__SERVICE_WAIT=300 NUMEXPR_MAX_THREADS=64 wandb agent $2"
  sleep 1
done
