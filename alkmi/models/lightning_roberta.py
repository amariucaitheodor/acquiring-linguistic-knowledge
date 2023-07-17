from typing import Any

from pytorch_lightning import LightningModule
from transformers import RobertaForMaskedLM, RobertaConfig
from transformers.modeling_outputs import MaskedLMOutput

from models.lightning_flava import configure_default_optimizers


class RobertaPreTrainingLightningModule(LightningModule):
    def __init__(self, **kwargs: Any):
        super().__init__()
        if kwargs['pretrained']:
            self.model = RobertaForMaskedLM.from_pretrained(kwargs['pretrained'])
            if kwargs['half_size']:
                print("WARNING: Loading pretrained RobertaForMaskedLM, so half_size is ignored.")
        else:
            # the official configuration sets the wrong values...
            self.model = RobertaForMaskedLM(RobertaConfig(vocab_size=50265,
                                                          max_position_embeddings=514,
                                                          num_hidden_layers=6 if kwargs['half_size'] else 12,
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
