import os
from datetime import timedelta

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
from lightning_models import BERTPreTrainingLightningModule, FlavaPreTrainingLightningModule, \
    RobertaPreTrainingLightningModule
from utils import build_config, update_ckt_dir_and_batch_size, assign_huggingface_ram, \
    initialize_multidatamodule, overwrite_config, build_model_kwargs


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

    config: AblationArguments = build_config()

    if config.training.use_wandb:
        wandb_logger = WandbLogger(
            project=f'alkmi-{config.datasets.ablation.train[0].key.split("/")[-1]}',
            log_model=False,  # set to "True" to also log checkpoints to WandB
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
        overwrite_config(struct=config.training, params=["learning_rate", "learning_rate_text_submodel",
                                                         "warmup_steps", "seed"])
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

    print(f"Building model '{config.model.name}'")
    if config.model.name == 'bert':
        model = BERTPreTrainingLightningModule(**build_model_kwargs(config.training, config.model))
        model = torch.compile(model)
    elif config.model.name == 'roberta':
        model = RobertaPreTrainingLightningModule(**build_model_kwargs(config.training, config.model))
        model = torch.compile(model)
    elif config.model.name == 'flava':
        print(f"Enabling TensorFloat32 tensor cores for float32 matrix multiplication")
        torch.set_float32_matmul_precision('medium')
        model = FlavaPreTrainingLightningModule(**build_model_kwargs(config.training, config.model))
    else:
        raise ValueError(f"Unknown model name: {config.model.name}")

    print("Registering basic callbacks")
    callbacks = [LearningRateMonitor(logging_interval="step"),
                 PseudoPerplexityCallback(key=config.datasets.ablation.val[0].key,
                                          split=config.datasets.ablation.val[0].split_key_mapping['validation'],
                                          limit_val_batches=config.training.lightning['limit_val_batches'],
                                          enable_progress_bar=config.training.lightning['enable_progress_bar']),
                 LMEvalHarnessCallback(enable_progress_bar=config.training.lightning['enable_progress_bar'])]

    if config.training.lightning_checkpoint is not None:
        callbacks.append(
            ModelCheckpoint(
                **OmegaConf.to_container(config.training.lightning_checkpoint),
                train_time_interval=timedelta(hours=3),
                monitor='evaluation/pseudo_perplexity',
                mode='min',
                save_top_k=-1,  # keep all checkpoints for later evaluation(s)
                save_last=True,
                verbose=True,
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

    print(f"Callbacks registered: {[type(c).__name__ for c in callbacks]}")

    trainer = Trainer(
        **OmegaConf.to_container(config.training.lightning),
        strategy=config.training.strategy,
        precision=config.training.precision,
        accelerator='gpu',
        max_time=timedelta(days=5),
        max_steps=450_000,
        num_sanity_val_steps=0,  # the cache is hit differently when starting without eval, so VRAM OOM is avoided
        inference_mode=False,  # conflicts with 2.0-compiled models
        gradient_clip_val=1.0,  # little effect on learning, but a "bad minibatch" could cause gradients to explode and
        # clipping prevents that iteration from disrupting the model
        callbacks=callbacks,
        logger=wandb_logger if config.training.use_wandb else True,
    )
    print(f"Trainer successfully initialized (strategy={type(trainer.strategy).__name__})")

    if config.training.use_wandb and trainer.global_rank == 0:
        wandb_logger.experiment.config.update(config)

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
