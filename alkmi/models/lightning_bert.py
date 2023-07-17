from typing import Any

from pytorch_lightning import LightningModule
from transformers import BertForMaskedLM, BertConfig
from transformers.modeling_outputs import MaskedLMOutput

from models.utils import configure_default_optimizers


class BERTPreTrainingLightningModule(LightningModule):
    def __init__(self, **kwargs: Any):
        super().__init__()
        if kwargs['pretrained']:
            self.model = BertForMaskedLM.from_pretrained(kwargs['pretrained'])
            if kwargs['half_size']:
                print("WARNING: Loading pretrained BertForMaskedLM, so half_size is ignored.")
        else:
            self.model = BertForMaskedLM(BertConfig(num_hidden_layers=6 if kwargs['half_size'] else 12))

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
