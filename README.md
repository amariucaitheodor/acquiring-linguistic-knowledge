# alkmi

Master's thesis of Theodor Amariucai, supervised by Alexander Warstadt and Prof. Ryan Cotterell.

![alt text](./assets/alkmi-low-resolution-color-logo.png "ALKMI logo")

### Introduction

### Project sections

1. [Text-vision pre-training](alkmi/README.md) works directly through [HuggingFace](https://huggingface.co/)
   and [WaB](https://wandb.ai/).

- Log in with *HuggingFace*: `huggingface-cli login`
- Log in with *Weights and Biases*: `wandb login`

```shell
poetry build
poetry install
poetry shell

cd alkmi/callbacks/lm-evaluation-harness
pip install -e ".[dev]"

# Test run:
bash run_locally.sh configs/debug.yaml

```

2. Evaluation involves the [lm-evaluation-harness](./lm-evaluation-harness/README.md) project
   for [BLiMP](https://github.com/alexwarstadt/blimp). Note: this requires >=`python3.9`.

### Miscellaneous

- For sweeps with WandB on the WiT configuration, run e.g.:

```shell
cd alkmi
wandb sweep configs/wit_final_sweep_config.yaml
bash run_sweep_on_cluster.sh NR_AGENTS SWEEP_ID
```

- To connect to the Euler compute node, first ssh into the login node then run e.g.:

```bash
# An already-running job
srun --interactive --jobid JOBID --pty bash
# Interactive node with 2 GPUs (20GBs VRAM each)
srun --gpus=2 \
  --gres=gpumem:20g \
  --ntasks-per-node=2 \
  --job-name "interactive" --cpus-per-task=4 --mem-per-cpu=15000 --nodes=1 --time=4:00:00 --pty --preserve-env $SHELL
# Interactive node with 1 GPU and a lot of RAM for the (initial) WiT collapsing
srun --gpus=1 \
  --gres=gpumem:20g \
  --ntasks-per-node=1 \
  --job-name "interactive" --cpus-per-task=4 --mem-per-cpu=15000 --nodes=1 --time=4:00:00 --pty --preserve-env $SHELL
# Processing node
srun --time=24:00:00 --ntasks-per-node=1 --cpus-per-task=16 --mem-per-cpu=16000 --nodes=1 --pty --preserve-env $SHELL

```

### Euler

- Check space:
```shell
du -h -d 2 ~ | sort -h                                   # home directory
du -h -d 2 /cluster/scratch/tamariucai/ | sort -h        # scratch space
du -h -d 2 /cluster/work/cotterell/tamariucai/ | sort -h # work directory
```

- For faster access, set up a `~/init.sh`:

```bash
env2lmod
module purge
module load eth_proxy gcc/8.2.0 python_gpu/3.10.4
export PATH="$HOME/.local/bin:$PATH"
export HF_DATASETS_CACHE="/cluster/scratch/tamariucai/HuggingfaceDatasets"
export HF_HOME="/cluster/work/cotterell/tamariucai/HuggingfaceHome"
export WANDB_CACHE_DIR="/cluster/scratch/tamariucai/WandbCache"
export WANDB_DIR="/cluster/work/cotterell/tamariucai/WandbDir"
export PYTHONPATH=/cluster/work/cotterell/tamariucai/acquiring-linguistic-knowledge/:/cluster/work/cotterell/tamariucai/acquiring-linguistic-knowledge/alkmi/callbacks/lm-evaluation-harness
cd /cluster/work/cotterell/tamariucai/acquiring-linguistic-knowledge
poetry shell
cd alkmi
```

### Submodules

To work with [submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules):

- Pull library updates with `git submodule update --remote --rebase`. This might also
  help `git submodule update --init --force --remote`.
- Push changes (including to the library) with `git push --recurse-submodules=check`
