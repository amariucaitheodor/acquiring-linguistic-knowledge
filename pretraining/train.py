import wandb
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from callbacks.blimp_eval import BlimpEvalCallback
from callbacks.imagenet_zeroshot_eval import MultimodalEvalCallback
from data.datamodules import ImagenetEvalDataModule
from definitions import FLAVAArguments
from model import BERTPreTrainingLightningModule, FlavaPreTrainingLightningModule
from utils import build_config, update_ckt_dir_and_batch_size, assign_huggingface_ram, \
    initialize_multidatamodule, overwrite_config, build_datamodule_kwargs


def main():
    config: FLAVAArguments = build_config()

    if config.training.use_wandb:
        wandb_logger = WandbLogger(
            project=f'flava-textvision-multimodal-{config.datasets.vl.train[0].key.split("/")[-1]}',
            log_model="all",  # checkpoints are logged during training
            tags=["pretrained" if config.model.pretrained else "scratch"],
            magic=True,
            force=True,
            save_code=True,
            entity='rycolab'
        )

        # Overwriting needs to be called after wandb.init()
        overwrite_config(struct=config.training.lightning, params=["accumulate_grad_batches"])
        overwrite_config(struct=config.training, params=["learning_rate", "warmup_steps", "seed"])
        overwrite_config(struct=config, params=["text_perc", "vision_perc"])
        update_ckt_dir_and_batch_size(config)

        wandb.run.tags += (f"{config.text_perc}% text",)
        wandb.run.tags += (f"{config.vision_perc}% vision",)
        wandb.run.tags += (config.model.name,)

    if config.training.seed != -1:
        seed_everything(config.training.seed, workers=True)

    print("Assigning HuggingFace RAM")
    assign_huggingface_ram()

    print("Initializing multidatamodule")
    datamodule = initialize_multidatamodule(config)

    print("Building model")
    if config.model.name == 'bert':
        model = BERTPreTrainingLightningModule(**config.model)
    elif config.model.name == 'flava':
        model = FlavaPreTrainingLightningModule(**config.model)
    else:
        raise ValueError(f"Unknown model name: {config.model.name}")

    print("Registering callbacks")
    callbacks = [LearningRateMonitor(logging_interval="step"), BlimpEvalCallback()]

    if config.datasets.imagenet is not None and config.model.name == 'flava':
        imagenet_validation_module = ImagenetEvalDataModule(
            **build_datamodule_kwargs(config.datasets.imagenet, config.training),
            name="ImageNetEvalDataModule"
        )
        imagenet_validation_module.setup(stage="validate")
        callbacks.append(MultimodalEvalCallback(imagenet_datamodule=imagenet_validation_module))

    if config.training.lightning_checkpoint is not None:
        callbacks.append(
            ModelCheckpoint(
                **OmegaConf.to_container(config.training.lightning_checkpoint)
            )
        )
    print(f"Callbacks registered: {callbacks}")

    if config.training.use_wandb:
        wandb_logger.experiment.config.update(config)

    print("Initializing trainer")
    trainer = Trainer(
        **OmegaConf.to_container(config.training.lightning),
        callbacks=callbacks,
        logger=wandb_logger if config.training.use_wandb else True,
    )

    print("Starting training")
    trainer.fit(model, datamodule=datamodule, ckpt_path=config.training.lightning_load_from_checkpoint)

    print("Starting validation")
    trainer.validate(model, datamodule=datamodule)

    if config.training.use_wandb:
        wandb.finish()  # [optional] finish the wandb run, necessary in notebooks


if __name__ == "__main__":
    main()
