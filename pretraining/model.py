from typing import Tuple, Any

import torch
from pytorch_lightning import LightningModule
from transformers import FlavaForPreTraining, FlavaConfig, BertForMaskedLM, BertConfig, get_cosine_schedule_with_warmup
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.flava.modeling_flava import FlavaForPreTrainingOutput

from utils import get_local_ram


class FlavaPreTrainingLightningModule(LightningModule):
    def __init__(self,
                 learning_rate: float = 0.0002,
                 adam_eps: float = 1.0e-08,
                 adam_weight_decay: float = 0.01,
                 adam_betas: Tuple[float, float] = (0.9, 0.999),
                 warmup_steps: int = 2000,
                 max_steps: int = 450000,
                 **kwargs: Any):
        super().__init__()
        if 'pretrained' in kwargs and kwargs['pretrained']:
            self.model = FlavaForPreTraining.from_pretrained("facebook/flava-full")
        else:
            self.model = FlavaForPreTraining(FlavaConfig(**kwargs))

        self.learning_rate = learning_rate
        self.adam_eps = adam_eps
        self.adam_betas = adam_betas
        self.adam_weight_decay = adam_weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def training_step(self, batch, batch_idx):
        output = self._step(batch)
        losses = output.loss_info
        total_loss = 0
        for key in losses:
            total_loss += losses[key]
            self.log(f"train/losses/{key}_loss", losses[key], prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        output = self._step(batch)
        losses = output.loss_info
        total_loss = 0
        for key in losses:
            total_loss += losses[key]
            self.log(f"validation/losses/{key}_loss", losses[key], prog_bar=True, logger=True, sync_dist=True)
        return total_loss

    def _step(self, batch) -> FlavaForPreTrainingOutput:
        return self.model(**batch, skip_unmasked_multimodal_encoder=True, return_loss=True)

    def configure_optimizers(self):
        return configure_default_optimizers(self.model, self.learning_rate, self.adam_eps, self.adam_weight_decay,
                                            self.adam_betas, self.warmup_steps, self.max_steps)


class BERTPreTrainingLightningModule(LightningModule):
    def __init__(self,
                 learning_rate: float = 0.0002,
                 adam_eps: float = 1.0e-08,
                 adam_weight_decay: float = 0.01,
                 adam_betas: Tuple[float, float] = (0.9, 0.999),
                 warmup_steps: int = 2000,
                 max_steps: int = 450000,
                 **kwargs: Any):
        super().__init__()
        if 'pretrained' in kwargs and kwargs['pretrained']:
            self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        else:
            self.model = BertForMaskedLM(BertConfig(**kwargs))

        self.learning_rate = learning_rate
        self.adam_eps = adam_eps
        self.adam_betas = adam_betas
        self.adam_weight_decay = adam_weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def training_step(self, batch, batch_idx):
        output = self._step(batch)
        self.log(f"train/losses/mlm_loss", output.loss, prog_bar=True, logger=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self._step(batch)
        self.log(f"validation/losses/mlm_loss", output.loss, prog_bar=True, logger=True)
        return output.loss

    def _step(self, batch) -> MaskedLMOutput:
        """
        Reuses the same processor as Flava, which shouldn't be a problem...
        """
        return self.model(input_ids=batch.get("input_ids"), labels=batch.get("mlm_labels"), return_dict=True)

    def configure_optimizers(self):
        return configure_default_optimizers(self.model, self.learning_rate, self.adam_eps, self.adam_weight_decay,
                                            self.adam_betas, self.warmup_steps, self.max_steps)


def configure_default_optimizers(model: Any, learning_rate, adam_eps, adam_weight_decay,
                                 adam_betas, warmup_steps, max_steps):
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
