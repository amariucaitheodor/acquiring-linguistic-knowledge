from typing import Tuple, Any, Iterator

import torch
from pytorch_lightning import LightningModule
from torch.nn import Parameter
from transformers import BertForMaskedLM, BertConfig, get_cosine_schedule_with_warmup, \
    RobertaForMaskedLM, RobertaConfig
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.flava.modeling_flava import FlavaForPreTrainingOutput

from models.flava import FlavaForPreTraining, FlavaConfig


class FlavaPreTrainingLightningModule(LightningModule):
    def __init__(self, **kwargs: Any):
        super().__init__()
        if 'pretrained' in kwargs and kwargs['pretrained']:
            self.model = FlavaForPreTraining.from_pretrained(kwargs['pretrained'])
        else:
            self.model = FlavaForPreTraining(FlavaConfig(compile_submodels=True))

        if 'learning_rate_text_submodel' in kwargs:
            text_lr = kwargs.pop('learning_rate_text_submodel')

            print(f"FLAVA will use a different learning rate for its text submodel "
                  f"({text_lr}) compared to its other submodels "
                  f"({kwargs['learning_rate']})")

            for n, _ in self.model.flava.text_model.named_parameters():
                n += 'text_model.'

            self.optimizers = configure_default_optimizers(parameters=[
                {'params': [p for n, p in self.model.named_parameters() if 'text_model' not in n]},
                {'params': self.model.flava.text_model.parameters(), 'lr': text_lr}
            ], **kwargs)
        else:
            print(f"FLAVA will use a global learning rate of {kwargs['learning_rate']}")
            self.optimizers = configure_default_optimizers(parameters=self.model.parameters(), **kwargs)

    def training_step(self, batch, batch_idx):
        output = self._step(batch)
        losses = output.loss_info
        total_loss = 0
        for key in losses:
            total_loss += losses[key]
            self.log(f"train/losses/{key}_loss", losses[key], prog_bar=True, logger=True, sync_dist=True)
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
        return self.optimizers


class BERTPreTrainingLightningModule(LightningModule):
    def __init__(self, **kwargs: Any):
        super().__init__()
        if 'pretrained' in kwargs and kwargs['pretrained']:
            self.model = BertForMaskedLM.from_pretrained(kwargs['pretrained'])
        else:
            self.model = BertForMaskedLM(BertConfig())

        self.optimizers = configure_default_optimizers(parameters=self.model.parameters(), **kwargs)

    def training_step(self, batch, batch_idx):
        output = self._step(batch)
        self.log(f"train/losses/mlm_loss", output.loss, prog_bar=True, logger=True, sync_dist=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self._step(batch)
        self.log(f"validation/losses/mlm_loss", output.loss, prog_bar=True, logger=True, sync_dist=True)
        return output.loss

    def _step(self, batch) -> MaskedLMOutput:
        return self.model(input_ids=batch.get("input_ids_masked"), labels=batch.get("mlm_labels"), return_dict=True)

    def configure_optimizers(self):
        return self.optimizers


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

        self.optimizers = configure_default_optimizers(parameters=self.model.parameters(), **kwargs)

    def training_step(self, batch, batch_idx):
        output = self._step(batch)
        self.log(f"train/losses/mlm_loss", output.loss, prog_bar=True, logger=True, sync_dist=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self._step(batch)
        self.log(f"validation/losses/mlm_loss", output.loss, prog_bar=True, logger=True, sync_dist=True)
        return output.loss

    def _step(self, batch) -> MaskedLMOutput:
        return self.model(input_ids=batch.get("input_ids_masked"), labels=batch.get("mlm_labels"), return_dict=True)

    def configure_optimizers(self):
        return self.optimizers


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
