#!/bin/bash

# Job details
TIME=120:00:00 # HH:MM (default: 04:00, max: 240:00)
NUM_GPUS=1     # GPUs per node
NUM_CPUS=2     # Number of cores (default: 1)
CPU_RAM=16000  # RAM for each core (default: 1024)

echo "Number of agents: $1, GPUs per agent: 1, sweep name: $2"

# --mail-type=END,FAIL uncomment to get email notifications
# Submit job
for ((i = 1; i <= $1; i++)); do
  NR=$i
  sbatch --job-name="multimodal" \
    --time=$TIME \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=$NUM_CPUS \
    --tmp=100G \
    --mem-per-cpu=$CPU_RAM \
    --gpus=$NUM_GPUS \
    --gres=gpumem:20g \
    -o "wit_sweep_run_($2)_#${NR}_($(date "+%F-%T")).results" \
    --wrap="WANDB__SERVICE_WAIT=300 NUMEXPR_MAX_THREADS=64 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 wandb agent $2"
done
