# acquiring-linguistic-knowledge

Master's thesis of Theodor Amariucai, supervised by Alexander Warstadt and Prof. Ryan Cotterell.

### Introduction

### Project sections

1. [Text-vision pre-training](./pretraining/README.md) works directly through [HuggingFace](https://huggingface.co/)
   and [WaB](https://wandb.ai/).

- Log in with *HuggingFace*: `huggingface-cli login`
- Log in with *Weights and Biases*: `wandb login`

```shell
poetry build
poetry install
poetry shell

cd pretraining/callbacks/lm-evaluation-harness
pip install -e ".[dev]"

# Test run:
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python -m train config=configs/debug.yaml

```

2. [Text-audio development](./audio/README.md) for the novel text-audio fusion model.
3. Evaluation involves the [lm-evaluation-harness](./lm-evaluation-harness/README.md) project
   for [BLiMP](https://github.com/alexwarstadt/blimp) and [...] for sBLiMP.

### Miscellaneous

Note: [BLiMP evaluation](./lm-evaluation-harness/README.md) requires >=`python3.9`.

- For sweeps with WandB on the WiT configuration, run e.g.:

```shell
cd examples
wandb sweep configs/wit_sweep_config.yaml
bash run_sweep_on_cluster.sh NR_AGENTS SWEEP_ID
```

- To connect to the Euler compute node, first ssh into the login node then run e.g.:

```bash
# An already-running job
srun --interactive --jobid JOBID --pty bash
# Interactive GPU node (N.B. don't 'source init.sh' again!)
srun --time=4:00:00 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=2 \
    --mem-per-cpu=30000 \
    --gpus=1 \
    --gres=gpumem:20g \
    --pty \
    --preserve-env \
    $SHELL
# Processing node
srun --time=24:00:00 --ntasks=1 --mem-per-cpu=256000 --nodes=1 --pty --preserve-env $SHELL
```

- If *WandB* isn't syncing, try running the following **on the compute node**:

```bash
pip install wandb --upgrade
```

### Euler

- For faster access, set up a `~/init.sh`:

```bash
env2lmod
module load eth_proxy gcc/8.2.0 python_gpu/3.10.4
export HF_DATASETS_CACHE="/cluster/scratch/tamariucai/HuggingfaceDatasets"
export HF_HOME="/cluster/work/cotterell/tamariucai/HuggingfaceHome"
export WANDB_CACHE_DIR="/cluster/scratch/tamariucai/WandbCache"
export WANDB_DIR="/cluster/work/cotterell/tamariucai/WandbDir"
export PYTHONPATH=/cluster/work/cotterell/tamariucai/acquiring-linguistic-knowledge/:/cluster/work/cotterell/tamariucai/acquiring-linguistic-knowledge/pretraining/callbacks/lm-evaluation-harness
cd /cluster/work/cotterell/tamariucai/acquiring-linguistic-knowledge/pretraining/
poetry shell
```

### Submodules

To work with [submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules):

- Pull library updates with `git submodule update --remote --rebase`. This might also
  help `git submodule update --init --force --remote`.
- Push changes (including to the library) with `git push --recurse-submodules=check`
