# acquiring-linguistic-knowledge

Master's thesis of Theodor Amariucai, supervised by Alexander Warstadt and Prof. Ryan Cotterell.

### Introduction

### Project sections

1. [Text-vision pre-training](./pretraining/README.md) works directly through [HuggingFace](https://huggingface.co/)
   and [WaB](https://wandb.ai/) for a smooth experience.

```shell
python -m venv .venv/acquiring-linguistic-knowledge
source .venv/acquiring-linguistic-knowledge/bin/activate
pip install -r requirements.txt

cd lm-evaluation-harness
pip install -e ".[dev]"

# Until issue is fixed (https://github.com/huggingface/transformers/issues/23047), manually change FLAVA source code:
nano /cluster/work/cotterell/tamariucai/acquiring-linguistic-knowledge/.venv/acquiring-linguistic-knowledge/lib64/python3.10/site-packages/transformers/models/flava/modeling_flava.py
# Rename all_gather_with_backprop to all_gather
```

2. [Text-audio development](./audio/README.md) for the novel text-audio fusion model.
3. Evaluation involves the [lm-evaluation-harness](./lm-evaluation-harness/README.md) project
   for [BLiMP](https://github.com/alexwarstadt/blimp) and [...] for sBLiMP.

### Miscellaneous

Note: [BLiMP evaluation](./lm-evaluation-harness/README.md) requires >=`python3.9`.

- To start a test run:

```python3
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python train.py config=configs/debug.yaml
```

- For sweeps with WandB on the debug configuration, run:

```bash
cd examples
wandb sweep flava/configs/pretraining/debug_sweep_config.yaml
# now run sweep agent according to instructions
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
    --tmp=400G \
    --mem-per-cpu=77000 \
    --gpus=1 \
    --gres=gpumem:20g \
    --pty \
    --preserve-env \
    $SHELL
# Processing node
srun --time=24:00:00 --tmp=650G --ntasks=1 --mem-per-cpu=256000 --nodes=1 --pty --preserve-env $SHELL
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
source /cluster/work/cotterell/tamariucai/acquiring-linguistic-knowledge/.venv/acquiring-linguistic-knowledge/bin/activate
export HF_DATASETS_CACHE="/cluster/scratch/tamariucai/HuggingfaceDatasets"
export HF_HOME="/cluster/work/cotterell/tamariucai/HuggingfaceHome"
export WANDB_CACHE_DIR="/cluster/scratch/tamariucai/WandbCache"
export PYTHONPATH=/cluster/work/cotterell/tamariucai/acquiring-linguistic-knowledge/:/cluster/work/cotterell/tamariucai/acquiring-linguistic-knowledge/lm-evaluation-harness
cd /cluster/work/cotterell/tamariucai/acquiring-linguistic-knowledge/scripts/
```

### Submodules

To work with [submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules):

- Pull library updates with `git submodule update --remote --rebase`. This might also
  help `git submodule update --init --force --remote`.
- Push changes (including to the library) with `git push --recurse-submodules=check`
