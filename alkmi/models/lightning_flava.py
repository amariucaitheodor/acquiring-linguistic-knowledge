from typing import Any

import warnings
from pytorch_lightning import LightningModule

from alkmi.models.flava.modeling_flava import FlavaForPreTrainingOutput
from alkmi.models.flava import FlavaForPreTraining, FlavaConfig
from models.utils import configure_default_optimizers


class FlavaPreTrainingLightningModule(LightningModule):
    def __init__(self, **kwargs: Any):
        super().__init__()
        if kwargs['pretrained']:
            self.model = FlavaForPreTraining.from_pretrained(kwargs['pretrained'])
            if kwargs['half_size']:
                print("WARNING: Loading pretrained FlavaForPreTraining, so half_size is ignored.")
        else:
            if not kwargs['half_size']:
                self.model = FlavaForPreTraining(FlavaConfig(compile_submodels=True))
            else:
                text_config = {
                    "hidden_size": 516,  # multiple of the number of attention heads (12)
                    "num_hidden_layers": 6 if kwargs['half_size'] else 12,
                }
                multimodal_config = {
                    "hidden_size": 516,  # multiple of the number of attention heads (12)
                    "num_hidden_layers": 6 if kwargs['half_size'] else 12,
                }
                image_config = {
                    "hidden_size": 516,  # multiple of the number of attention heads (12)
                    "num_hidden_layers": 6 if kwargs['half_size'] else 12,
                }
                self.model = FlavaForPreTraining(FlavaConfig(compile_submodels=True,
                                                             hidden_size=516,
                                                             projection_dim=516,
                                                             image_config=image_config,
                                                             text_config=text_config,
                                                             multimodal_config=multimodal_config))

        # We downscale the loss as patience exhausts, so we need to readjust for logging
        self.original_weights = {
            "global_contrastive": self.model.global_contrastive_weight,
            "mmm_image": self.model.mmm_image_weight,
            "mmm_text": self.model.mmm_text_weight,
            "mlm": self.model.mlm_weight,
            "mim": self.model.mim_weight,
            "itm": self.model.itm_weight,
        }

        if 'learning_rate_text_submodel' in kwargs and kwargs['learning_rate_text_submodel']:
            if 'precision' in kwargs and kwargs['precision'] == "16-mixed":
                warnings.warn("Precision 16-mixed doesn't work well with LR parameter sets! Reverting...")
                print(f"FLAVA will use a global learning rate of {kwargs['learning_rate']}")
                self.optimizers = configure_default_optimizers(parameters=self.model.parameters(), **kwargs)
            else:
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
        for key in losses:
            upscale_factor = self.original_weights[key] / self.model.__getattribute__(f"{key}_weight")
            self.log(f"train/losses/{key}_loss", losses[key] * upscale_factor,
                     prog_bar=True, logger=True, sync_dist=True)
        return output.loss  # total loss

    def validation_step(self, batch, batch_idx):
        output = self._step(batch)
        losses = output.loss_info
        for key in losses:
            upscale_factor = self.original_weights[key] / self.model.__getattribute__(f"{key}_weight")
            self.log(f"validation/losses/{key}_loss", losses[key] * upscale_factor,
                     prog_bar=True, logger=True, sync_dist=True)
        return output.loss  # total loss

    def _step(self, batch) -> FlavaForPreTrainingOutput:
        return self.model(**batch, skip_unmasked_multimodal_encoder=True, return_loss=True)

    def configure_optimizers(self):
        return self.optimizers
