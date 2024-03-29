#!/bin/bash

# Job details
TIME=5-00:00:00 # HH:MM:SS (default: 04:00, max: 240:00)
NUM_CPUS=8   # Number of cores (default: 1)
CPU_RAM=64000  # LESS THAN THIS MIGHT OOM_KILL THE JOB! RAM for each core (default: 1024)

# Load modules (might want to run 'env2lmod' prior to this)
module load eth_proxy

# Submit job
sbatch --mail-type=END,FAIL \
  --job-name="process_wit" \
  --time=$TIME \
  --ntasks-per-node=1 \
  --cpus-per-task=$NUM_CPUS \
  --tmp=350G \
  --mem-per-cpu=$CPU_RAM \
  -o "initial_setup_$(date "+%F-%T").log" \
  --wrap="NUMEXPR_MAX_THREADS=164 python initial_setup.py"
