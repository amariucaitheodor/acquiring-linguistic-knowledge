#!/bin/bash

RESUME_ID=${1-}
WANDB_RUN_GROUP=${2-"text1-vision1"}
NUM_GPUS=${3-1}
VRAM_PER_GPU=${4-40g}
CONFIG="wit_resume_model.yaml"
echo "Selected configuration: configs/flava/$CONFIG, GPUs: $NUM_GPUS, $VRAM_PER_GPU VRAM/GPU, resuming run $RESUME_ID (ID) in group '$WANDB_RUN_GROUP'"

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
  --output "resume_singlenode_$(date "+%F-%T").log" \
  --wrap="WANDB_RUN_GROUP=$WANDB_RUN_GROUP WANDB__SERVICE_WAIT=300 WANDB_RESUME=must WANDB_RUN_ID=$RESUME_ID NUMEXPR_MAX_THREADS=64 python -m train config=configs/flava/$CONFIG"

# eu-a65-03 might have an issue
# --nodelist=eu-a65-02,eu-a65-04,eu-a65-05,eu-a65-07 \
