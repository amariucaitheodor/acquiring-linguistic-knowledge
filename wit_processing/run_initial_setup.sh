#!/bin/bash

# Job details
TIME=96:00:00 # HH:MM:SS (default: 04:00, max: 240:00)
NUM_CPUS=12   # Number of cores (default: 1)
CPU_RAM=72000  # RAM for each core (default: 1024)

# Load modules (might want to run 'env2lmod' prior to this)
module load eth_proxy

# Submit job
sbatch --mail-type=END,FAIL \
  --job-name="process_wit" \
  --time=$TIME \
  --ntasks-per-node=1 \
  --cpus-per-task=$NUM_CPUS \
  --tmp=650G \
  --mem-per-cpu=$CPU_RAM \
  -o "process_run_$(date "+%F-%T").results" \
  --wrap="NUMEXPR_MAX_THREADS=164 python initial_setup.py"
