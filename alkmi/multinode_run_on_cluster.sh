#!/bin/bash

NODES=${2-1}
VRAM_PER_GPU=${3-20g}
echo "Selected configuration: $1, nodes: $NODES (1 GPU/node, $VRAM_PER_GPU VRAM/GPU)"

# --mail-type=END,FAIL uncomment to get email notifications
# --nodes needs to match Trainer(num_nodes=...)
# --ntasks-per-node needs to match Trainer(devices=...)
sbatch --job-name="multimodal" \
  --time=5-00:00:00 \
  --nodes="$NODES" \
  --ntasks="$NODES" \
  --ntasks-per-node=1 \
  --gpus-per-node=1 \
  --cpus-per-task=2 \
  --tmp=100G \
  --mem-per-cpu=8000 \
  --gres=gpumem:"$VRAM_PER_GPU" \
  -o "multinode_run_$(date "+%F-%T").results" \
  --wrap="WANDB_RUN_GROUP=DDP-$(date "+%F-%T") WANDB__SERVICE_WAIT=300 NUMEXPR_MAX_THREADS=64 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 srun python -m train config=$1"
