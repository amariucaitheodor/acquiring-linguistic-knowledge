from typing import Tuple, Any

import torch
from pytorch_lightning import LightningModule
from transformers import FlavaConfig, BertForMaskedLM, BertConfig, get_cosine_schedule_with_warmup, \
    RobertaForMaskedLM, RobertaConfig
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.flava.modeling_flava import FlavaForPreTrainingOutput

from models.flava import FlavaForPreTraining


class FlavaPreTrainingLightningModule(LightningModule):
    def __init__(self, **kwargs: Any):
        super().__init__()
        if 'pretrained' in kwargs and kwargs['pretrained']:
            self.model = FlavaForPreTraining.from_pretrained(kwargs['pretrained'])
        else:
            self.model = FlavaForPreTraining(FlavaConfig())

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
        for key in losses:
            self.log(f"validation/losses/{key}_loss", losses[key], prog_bar=True, logger=True, sync_dist=True)
        return output.loss  # total loss

    def _step(self, batch) -> FlavaForPreTrainingOutput:
        return self.model(**batch, skip_unmasked_multimodal_encoder=True, return_loss=True)

    def configure_optimizers(self):
        return configure_default_optimizers(
            self.model,
            learning_rate=2e-4,
            adam_eps=1e-8,
            adam_weight_decay=1e-2,
            adam_betas=(0.9, 0.999),
            warmup_steps=2000,
            max_steps=450000,
        )


class BERTPreTrainingLightningModule(LightningModule):
    def __init__(self, **kwargs: Any):
        super().__init__()
        if 'pretrained' in kwargs and kwargs['pretrained']:
            self.model = BertForMaskedLM.from_pretrained(kwargs['pretrained'])
        else:
            self.model = BertForMaskedLM(BertConfig())

    def training_step(self, batch, batch_idx):
        output = self._step(batch)
        self.log(f"train/losses/mlm_loss", output.loss, prog_bar=True, logger=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self._step(batch)
        self.log(f"validation/losses/mlm_loss", output.loss, prog_bar=True, logger=True)
        return output.loss

    def _step(self, batch) -> MaskedLMOutput:
        return self.model(input_ids=batch.get("input_ids_masked"), labels=batch.get("mlm_labels"), return_dict=True)

    def configure_optimizers(self):
        return configure_default_optimizers(
            self.model,
            learning_rate=2e-4,
            adam_eps=1e-8,
            adam_weight_decay=1e-2,
            adam_betas=(0.9, 0.999),
            warmup_steps=2000,
            max_steps=450000,
        )


class RobertaPreTrainingLightningModule(LightningModule):
    def __init__(self, **kwargs: Any):
        super().__init__()
        if 'pretrained' in kwargs and kwargs['pretrained']:
            self.model = RobertaForMaskedLM.from_pretrained(kwargs['pretrained'])
        else:
            # the official configuration sets the wrong values...
            self.model = RobertaForMaskedLM(RobertaConfig(vocab_size=50265,
                                                          max_position_embeddings=514,
                                                          layer_norm_eps=1e-05,
                                                          type_vocab_size=1,
                                                          ))

    def training_step(self, batch, batch_idx):
        output = self._step(batch)
        self.log(f"train/losses/mlm_loss", output.loss, prog_bar=True, logger=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self._step(batch)
        self.log(f"validation/losses/mlm_loss", output.loss, prog_bar=True, logger=True)
        return output.loss

    def _step(self, batch) -> MaskedLMOutput:
        return self.model(input_ids=batch.get("input_ids_masked"), labels=batch.get("mlm_labels"), return_dict=True)

    def configure_optimizers(self):
        return configure_default_optimizers(
            self.model,
            learning_rate=2e-4,
            adam_eps=1e-8,
            adam_weight_decay=1e-2,
            adam_betas=(0.9, 0.999),
            warmup_steps=2000,
            max_steps=450000,
        )


def configure_default_optimizers(
        model: torch.nn.Module,
        learning_rate: float,
        adam_eps: float,
        adam_weight_decay: float,
        adam_betas: Tuple[float, float],
        warmup_steps: int,
        max_steps: int):
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
