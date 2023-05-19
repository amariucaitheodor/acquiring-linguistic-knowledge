import torch

import wandb
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from callbacks.blimp_eval import LMEvalHarnessCallback
from callbacks.multimodal_overfitting_monitor import MultimodalOverfittingMonitor
from callbacks.pseudo_perplexity_eval import PseudoPerplexityCallback
from definitions import AblationArguments
from lightning_models import BERTPreTrainingLightningModule, FlavaPreTrainingLightningModule, RobertaPreTrainingLightningModule
from utils import build_config, update_ckt_dir_and_batch_size, assign_huggingface_ram, \
    initialize_multidatamodule, overwrite_config, build_model_kwargs


def main():
    config: AblationArguments = build_config()

    if config.training.use_wandb:
        wandb_logger = WandbLogger(
            project=f'multimodal-{config.datasets.ablation.train[0].key.split("/")[-1]}',
            resume=False,
            log_model=True,  # checkpoints are logged at the end of training
            tags=[config.model.pretrained if config.model.pretrained else "scratch"],
            magic=True,
            force=True,
            save_code=True,
            entity='rycolab'
        )

        wandb.init(**wandb_logger._wandb_init)

        # Overwriting needs to be called after wandb.init()
        overwrite_config(struct=config.model, params=["mlm_perc"])
        overwrite_config(struct=config, params=["text_perc", "vision_perc"])
        overwrite_config(struct=config.training.lightning, params=["accumulate_grad_batches"])
        overwrite_config(struct=config.training, params=["learning_rate", "warmup_steps", "seed"])
        overwrite_config(struct=config.datasets, params=["sampling_temperature"])
        update_ckt_dir_and_batch_size(config)

        wandb.run.tags += (f"{config.text_perc}% text",)
        wandb.run.tags += (f"{config.vision_perc}% vision",)
        wandb.run.tags += (config.model.name,)

    if config.training.seed != -1:
        seed_everything(config.training.seed, workers=True)

    # IMPORTANT KNOB!
    if config.text_perc >= config.vision_perc:
        print(f"Text is the predominant modality ({config.text_perc} v.s. {config.vision_perc} vision), "
              f"will sample proportionally for optimal BLiMP performance.")
        config.datasets.sampling_temperature = 1.
    else:
        print(f"Text isn't the predominant modality ({config.text_perc} v.s. {config.vision_perc} vision), "
              f"will over-sample text (uniform rates) for better BLiMP performance.")
        config.datasets.sampling_temperature = 0.

    print("Assigning HuggingFace RAM")
    assign_huggingface_ram()

    print("Building model")
    torch._dynamo.config.cache_size_limit = 96
    if config.model.name == 'bert':
        model = BERTPreTrainingLightningModule(**build_model_kwargs(config.training, config.model))
        model = torch.compile(model)
    elif config.model.name == 'roberta':
        model = RobertaPreTrainingLightningModule(**build_model_kwargs(config.training, config.model))
        model = torch.compile(model)
    elif config.model.name == 'flava':
        model = FlavaPreTrainingLightningModule(**build_model_kwargs(config.training, config.model))
    else:
        raise ValueError(f"Unknown model name: {config.model.name}")

    print("Registering basic callbacks")
    callbacks = [LearningRateMonitor(logging_interval="step"),
                 LMEvalHarnessCallback(),
                 PseudoPerplexityCallback(key=config.datasets.ablation.val[0].key,
                                          limit_val_batches=config.training.lightning['limit_val_batches'])]

    if config.training.lightning_checkpoint is not None:
        callbacks.append(
            ModelCheckpoint(
                **OmegaConf.to_container(config.training.lightning_checkpoint)
            )
        )

    if config.text_perc + config.vision_perc > 0:
        print("Initializing datamodule")
        datamodule = initialize_multidatamodule(config)

        def add_monitor(name: str):
            nonlocal callbacks
            callbacks.append(MultimodalOverfittingMonitor(monitor=f'validation/losses/{name}', datamodule=datamodule,
                                                          patience=2, verbose=True, strict=False))

        print("Registering multimodal callbacks (overfitting monitors)")
        if config.text_perc > 0:
            add_monitor(name="mlm_loss")
        if config.vision_perc > 0:
            add_monitor(name="mim_loss")
        if config.text_perc > 0 and config.vision_perc > 0:
            for val_loss in ["itm_loss", "global_contrastive_loss", "mmm_image_loss", "mmm_text_loss"]:
                add_monitor(name=val_loss)

    print(f"Callbacks registered: {callbacks}")

    if config.training.use_wandb:
        wandb_logger.experiment.config.update(config)

    print("Initializing trainer")
    trainer = Trainer(
        **OmegaConf.to_container(config.training.lightning),
        callbacks=callbacks,
        logger=wandb_logger if config.training.use_wandb else True,
    )

    if config.text_perc + config.vision_perc > 0:
        print("Starting training")
        trainer.fit(model, datamodule=datamodule, ckpt_path=config.training.lightning_load_from_checkpoint)
    else:
        # A datamodule MUST be passed for the trainer.validate() method to work. Validation loss is irrelevant.
        config.text_perc = 1
        datamodule = initialize_multidatamodule(config)

    print("Starting validation")
    trainer.validate(model, datamodule=datamodule)

    if config.training.use_wandb:
        wandb.finish()  # [optional] finish the wandb run, necessary in notebooks


if __name__ == "__main__":
    main()
