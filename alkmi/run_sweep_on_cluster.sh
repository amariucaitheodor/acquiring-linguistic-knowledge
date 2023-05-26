#!/bin/bash

VRAM_PER_GPU=${3-20g}
echo "Number of agents: $1, GPUs per agent: 1, sweep name: $2, $VRAM_PER_GPU VRAM/GPU"

# --mail-type=END,FAIL uncomment to get email notifications
# Submit job
for ((i = 1; i <= $1; i++)); do
  NR=$i
  sbatch --job-name="multimodal" \
    --time=5-00:00:00 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --gpus=1 \
    --cpus-per-task=2 \
    --tmp=100G \
    --mem-per-cpu=8000 \
    --gres=gpumem:"$VRAM_PER_GPU" \
    -o "sweep_run_($2)_#${NR}_($(date "+%F-%T")).results" \
    --wrap="WANDB__SERVICE_WAIT=300 NUMEXPR_MAX_THREADS=64 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 wandb agent $2"
done
