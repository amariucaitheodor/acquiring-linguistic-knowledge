from datetime import timedelta

import os
import torch

import wandb
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from definitions import AblationArguments
from lightning_models import BERTPreTrainingLightningModule, FlavaPreTrainingLightningModule, \
    RobertaPreTrainingLightningModule
from utils import build_config, update_ckt_dir_and_batch_size, assign_huggingface_ram, \
    initialize_multidatamodule, overwrite_config, build_model_kwargs


# N.B. Also set:
# limit_val_batches: 0
# accumulate_grad_batches: 1
# enable_progress_bar: true
# num_workers: 1
def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

    config: AblationArguments = build_config()

    if config.training.use_wandb:
        wandb_logger = WandbLogger(
            project=f'alkmi-{config.datasets.ablation.train[0].key.split("/")[-1]}',
            name="overfitting_run",
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
    callbacks = [LearningRateMonitor(logging_interval="step")]

    if config.text_perc + config.vision_perc > 0:
        print("Initializing datamodule")
        datamodule = initialize_multidatamodule(config)

    print(f"Callbacks registered: {[type(c).__name__ for c in callbacks]}")

    print("Initializing trainer")
    trainer = Trainer(
        devices=-1,
        log_every_n_steps=1,
        limit_val_batches=0,
        enable_checkpointing=False,
        accumulate_grad_batches=1,
        strategy="auto",
        enable_progress_bar=True,
        overfit_batches=1,
        accelerator='gpu',
        max_time=timedelta(days=5),
        max_epochs=-1,
        max_steps=-1,
        min_steps=45_000,
        precision="16-mixed",
        num_sanity_val_steps=0,  # the cache is hit differently when starting without eval (and VRAM OOM is avoided)
        inference_mode=False,  # conflicts with 2.0-compiled models
        gradient_clip_val=1.0,  # little effect on learning, but a "bad minibatch" could cause gradients to explode and
        # clipping prevents that iteration from disrupting the model
        callbacks=callbacks,
        logger=wandb_logger if config.training.use_wandb else True,
    )

    print(f"Strategy: {trainer.strategy}")

    if config.training.use_wandb and trainer.global_rank == 0:
        wandb_logger.experiment.config.update(config)

    print("Starting overfitting")
    trainer.fit(model, datamodule=datamodule)

    if config.training.use_wandb:
        wandb.finish()  # [optional] finish the wandb run, necessary in notebooks


if __name__ == "__main__":
    main()
