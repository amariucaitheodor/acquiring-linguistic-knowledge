#!/bin/bash

# Job details
TIME=120:00:00 # HH:MM (default: 04:00, max: 240:00)
NUM_GPUS=1    # GPUs per node
NUM_CPUS=2    # Number of cores (default: 1)
CPU_RAM=65000 # RAM for each core (default: 1024)

# Load modules (might want to run 'env2lmod' prior to this)
module load eth_proxy gcc/8.2.0 python_gpu/3.10.4

# Submit job
sbatch --mail-type=END,FAIL \
  --job-name="multimodal" \
  --time=$TIME \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=$NUM_CPUS \
  --tmp=550G \
  --mem-per-cpu=$CPU_RAM \
  --gpus=$NUM_GPUS \
  --gres=gpumem:40g \
  -o "wit_run_$(date "+%F-%T").results" \
  --wrap="WANDB__SERVICE_WAIT=300 NUMEXPR_MAX_THREADS=64 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python3.10 -m train config=configs/wit.yaml"
