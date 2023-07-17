from typing import Iterator, Tuple, Any

import torch
from torch.nn import Parameter
from transformers import get_cosine_schedule_with_warmup


def configure_default_optimizers(
        parameters: (list[dict[str, Iterator[Parameter]]] | Iterator[Parameter]),
        learning_rate: float,
        adam_eps: float,
        adam_weight_decay: float,
        adam_betas: Tuple[float, float],
        warmup_steps: int,
        max_steps: int,
        **kwargs: Any):
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        betas=adam_betas,
        eps=adam_eps,
        weight_decay=adam_weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
