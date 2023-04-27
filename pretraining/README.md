## Pretraining

- Set up the virtual environment and install the requirements:

```shell
python -m venv .venv/acquiring-linguistic-knowledge
source .venv/acquiring-linguistic-knowledge/bin/activate
pip install -r requirements.txt

cd lm-evaluation-harness
pip install -e ".[dev]"
```

- Log in with *HuggingFace*: `huggingface-cli login`
- Log in with *Weights and Biases*: `wandb login`