#!/bin/bash

# Job details
TIME=48:00:00 # HH:MM (default: 04:00, max: 240:00)
NUM_CPUS=12   # Number of cores (default: 1)
CPU_RAM=6000  # RAM for each core (default: 1024)

# Load modules (might want to run 'env2lmod' prior to this)
module load eth_proxy

# Submit job
sbatch --mail-type=END,FAIL \
  --job-name="word_counter_wit" \
  --time=$TIME \
  --ntasks-per-node=1 \
  --cpus-per-task=$NUM_CPUS \
  --tmp=650G \
  --mem-per-cpu=$CPU_RAM \
  -o "word_counter_$(date "+%F-%T").log" \
  --wrap="NUMEXPR_MAX_THREADS=164 python word_counter.py"
