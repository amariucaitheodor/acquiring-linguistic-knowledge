import os
from datetime import timedelta
import socket
from typing import List

import torch

import wandb
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from callbacks.blimp_eval import LMEvalHarnessCallback
from callbacks.imagenet_zeroshot import ImageNetZeroshotCallback
from callbacks.multimodal_scheduler import MultimodalScheduler
from callbacks.pseudo_perplexity_eval import PseudoPerplexityCallback
from definitions import AblationArguments
from models.lightning_bert import BERTPreTrainingLightningModule
from models.lightning_flava import FlavaPreTrainingLightningModule
from models.lightning_roberta import RobertaPreTrainingLightningModule
from utils import build_config, update_ckeckpoint_dir, initialize_multidatamodule, overwrite_config, build_model_kwargs, \
    using_single_large_gpu


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

    print(f"The current host is {socket.gethostname()}")
    print(f"The current cuDNN version is {torch.backends.cudnn.version()}")

    config: AblationArguments = build_config()

    if config.training.use_wandb:
        wandb_logger = WandbLogger(
            project=f'alkmi-{config.datasets.ablation.train[0].key.split("/")[-1]}',
            log_model=False,  # set to "True" to also log checkpoints to WandB
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

        if using_single_large_gpu(vram_threshold_gb=79):
            # 40 GB is the expected VRAM amount, but if we have more, we can double the batch size
            config.training.batch_size *= 2
            print(f"Detected 80 GB of RAM, doubling batch size to {config.training.batch_size}.")

            new_accumulation = max(1, config.training.lightning.get('accumulate_grad_batches') // 2)
            config.training.lightning.__setattr__("accumulate_grad_batches", new_accumulation)
            print(f"Detected 80 GB of RAM, halving gradient accumulation to "
                  f"{config.training.lightning.get('accumulate_grad_batches')}")
        elif torch.cuda.device_count() > 1:
            new_accumulation = max(1, config.training.lightning.get(
                'accumulate_grad_batches') // torch.cuda.device_count())
            config.training.lightning.__setattr__("accumulate_grad_batches", new_accumulation)
            print(f"Detected more than 1 GPU ({torch.cuda.device_count()}), decreasing gradient accumulation to "
                  f"{config.training.lightning.get('accumulate_grad_batches')}")

        if config.model.half_size:
            # we have (almost) twice as fewer parameters, we can (almost) double the batch size
            config.training.batch_size = int(config.training.batch_size * 1.5)
            print(f"Detected half-sized run, (almost) doubling batch size to {config.training.batch_size}.")

            new_accumulation = max(1, int(config.training.lightning.get('accumulate_grad_batches') / 1.5))
            config.training.lightning.__setattr__("accumulate_grad_batches", new_accumulation)
            print(f"Detected half-sized run, (almost) halving gradient accumulation to "
                  f"{config.training.lightning.get('accumulate_grad_batches')}")

        batch_size = config.training.batch_size * \
                     config.training.lightning.get('accumulate_grad_batches') * \
                     torch.cuda.device_count()

        update_ckeckpoint_dir(config, batch_size)

        wandb.run.tags += (f"{'half' if config.model.half_size else 'full'}-size",)
        wandb.run.tags += (f"{config.text_perc}% text",)
        wandb.run.tags += (f"{config.vision_perc}% vision",)
        wandb.run.tags += (f"bs{batch_size}",)
        wandb.run.tags += (f"seed{config.training.seed}",)
        wandb.run.tags += (config.training.precision,)
        if config.model.name != 'flava':
            wandb.run.tags += (config.model.name,)  # not default for our studies
        if config.model.pretrained:
            wandb.run.tags += ("pretrained",)  # not default for our studies

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

    print("Registering basic callbacks")
    callbacks: List = [LearningRateMonitor(logging_interval="step"),
                       PseudoPerplexityCallback(key=config.datasets.ablation.val[0].key,
                                                split=config.datasets.ablation.val[0].split_key_mapping['validation'],
                                                limit_val_batches=config.training.lightning['limit_val_batches'],
                                                enable_progress_bar=config.training.lightning['enable_progress_bar']),
                       LMEvalHarnessCallback(enable_progress_bar=config.training.lightning['enable_progress_bar'])]

    print(f"Building model '{config.model.name}'")
    if config.model.name == 'bert':
        model = BERTPreTrainingLightningModule(**build_model_kwargs(config.training, config.model))
        model = torch.compile(model)
    elif config.model.name == 'roberta':
        model = RobertaPreTrainingLightningModule(**build_model_kwargs(config.training, config.model))
        model = torch.compile(model)
    elif config.model.name == 'flava':
        print(f"Enabling TensorFloat32 tensor cores for float32 matrix multiplication")
        torch.set_float32_matmul_precision('high')
        model = FlavaPreTrainingLightningModule(**build_model_kwargs(config.training, config.model))
    else:
        raise ValueError(f"Unknown model name: {config.model.name}")

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
        print("Initializing multi-datamodule")
        datamodule = initialize_multidatamodule(config)

        def add_monitor(name: str, original_weight: float, patience: int):
            nonlocal callbacks
            callbacks.append(MultimodalScheduler(monitor=f'validation/losses/{name}',
                                                 datamodule=datamodule,
                                                 original_weight=original_weight,
                                                 load_prev_best_score=config.model.load_prev_best_score,
                                                 patience=patience))

        print("Registering multimodal overfitting monitors")
        # For MLM and MIM we directly reduce modality sampling weights (instead of task weights) to avoid burning cycles
        if config.text_perc > 0:
            mlm_weight = datamodule.sampling_weights[2 if len(datamodule.sampling_weights) > 1 else 0]
            add_monitor("mlm_loss", mlm_weight, patience=3)
        if config.vision_perc > 0:
            mim_weight = datamodule.sampling_weights[1 if len(datamodule.sampling_weights) > 1 else 0]
            add_monitor("mim_loss", mim_weight, patience=3)
        if config.text_perc > 0 and config.vision_perc > 0:  # shakier than the unimodal ones - need higher patience
            for objective in ['mmm_image', 'mmm_text', 'itm', 'global_contrastive']:
                add_monitor(f"{objective}_loss", model.model.__getattribute__(f"{objective}_weight"), patience=4)

            print("Adding ImageNet zeroshot callback")
            callbacks.append(ImageNetZeroshotCallback(config.training.lightning['enable_progress_bar']))

    print(f"Callbacks registered: {[type(c).__name__ for c in callbacks]}")

    trainer = Trainer(
        **OmegaConf.to_container(config.training.lightning),
        strategy=config.training.strategy,
        precision=config.training.precision,
        accelerator='gpu',
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
        should_allow_val_change = 'WANDB_RESUME' in os.environ and os.environ['WANDB_RESUME'] == 'must'
        wandb_logger.experiment.config.update(config, allow_val_change=should_allow_val_change)

    if config.text_perc + config.vision_perc > 0:
        print("Starting training")
        trainer.fit(model, datamodule=datamodule, ckpt_path=config.training.lightning_load_from_checkpoint)
    else:
        # A datamodule MUST be passed for the trainer.validate() method to work. Validation loss is irrelevant.
        config.text_perc = 1
        datamodule = initialize_multidatamodule(config)

    # Restore the original weights for evaluation (since no backprop, and it crashes if they're all 0).
    # Should also do this for the other models...
    if config.model.name == 'flava':
        for key, val in model.original_weights.items():
            model.__setattr__(f"{key}_weight", val)

    print("Starting validation")
    trainer.validate(model, datamodule=datamodule)

    if config.training.use_wandb:
        wandb.run.tags += ("completed",)
        wandb.finish()  # [optional] finish the wandb run, necessary in notebooks


if __name__ == "__main__":
    main()
