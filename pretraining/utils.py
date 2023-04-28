import os
import random
from typing import List

import datasets
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf

from data.datamodules import MLMDataModule, ImageDataModule, VLDataModule
from data.multidata import MultiDataModule
from pretraining.definitions import TrainingSingleDatasetInfo, TrainingArguments, FLAVAArguments


def build_datamodule_kwargs(
        dm_config: TrainingSingleDatasetInfo, training_config: TrainingArguments
):
    kwargs = {
        "train_infos": dm_config.train,
        "val_infos": dm_config.val,
        "batch_size": dm_config.batch_size or training_config.batch_size,
        "num_workers": dm_config.num_workers or training_config.num_workers,
        "allow_uneven_batches": dm_config.allow_uneven_batches,
    }
    kwargs.update(dm_config.datamodule_extra_kwargs)
    return kwargs


def build_config():
    cli_conf = OmegaConf.from_cli()
    if "config" not in cli_conf:
        raise ValueError(
            "Please pass 'config' to specify configuration yaml file for running FLAVA"
        )
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = instantiate(yaml_conf)
    cli_conf.pop("config")
    config: FLAVAArguments = OmegaConf.merge(conf, cli_conf)

    assert (
            "max_steps" in config.training.lightning
    ), "lightning config must specify 'max_steps'"

    assert 0 <= config.text_perc <= 100, "'text_perc' ablation study percentage must be between 0 and 100"
    assert 0 <= config.vision_perc <= 100, "'vision_perc' ablation study percentage must be between 0 and 100"

    if config.model.name == 'bert':
        assert config.vision_perc == 0, f"BERT can only be trained on text (vision_perc was {config.vision_perc})"

    return config


def initialize_multidatamodule(config: FLAVAArguments) -> MultiDataModule:
    """
    Suppose we are training on 100% of texts and 10% of images. We will sample the first 10% of the data (paired
    image and text) for the multimodal dataloader, the first 100% of the pairs for the text unimodal dataloader,
    and the first 10% for the vision dataloader. This means all images in this run will be paired with a text,
    but not all texts will be paired with an image. Nevertheless, there is still an MIM loss and duplication exists:
    the same images and text are seen both in the multimodal dataloader and in their corresponding unimodal dataloaders.

    Parameters
    ----------
    config : FLAVA arguments containing dataset configurations (text and vision percentages, what data to use, etc.)
    print_stats : Print all registered datasets, along with their sizes and other details.

    Returns a MultiDataModule set up for training.
    -------

    """
    modules = []

    multimodal_perc = min(config.vision_perc, config.text_perc)
    if multimodal_perc > 0:
        vl_config = copy_dataset_config_with_training_subset(config.datasets.ablation, percentage=multimodal_perc)
        vl_datamodule = VLDataModule(
            **build_datamodule_kwargs(vl_config, config.training),
            mlm_probability=0.4,  # https://arxiv.org/abs/2202.08005
            name="VLDataModule"
        )
        modules.append(vl_datamodule)

    if config.vision_perc > 0:
        vision_config = copy_dataset_config_with_training_subset(config.datasets.ablation,
                                                                 percentage=config.vision_perc)
        vision_datamodule = ImageDataModule(
            **build_datamodule_kwargs(vision_config, config.training),
            name="ImageDataModule"
        )
        modules.append(vision_datamodule)

    if config.text_perc > 0:
        text_config = copy_dataset_config_with_training_subset(config.datasets.ablation, percentage=config.text_perc)

        mlm_datamodule = MLMDataModule(
            **build_datamodule_kwargs(text_config, config.training),
            mlm_probability=0.4,  # https://arxiv.org/abs/2202.08005
            name="MLMDataModule"
        )
        modules.append(mlm_datamodule)

    if len(modules) == 3:
        sampling_weights = [0.75, 0.15, 0.15]
    else:
        assert len(modules) == 1
        sampling_weights = [1.]

    datamodule = MultiDataModule(datamodules=modules,
                                 # Table A.2 in https://arxiv.org/pdf/2112.04482.pdf
                                 sampling_func=lambda: random.choices(population=range(len(modules)),
                                                                      weights=sampling_weights, k=1)[0])
    datamodule.setup("fit")
    return datamodule


def overwrite_config(struct, params: List[str]):
    for h in params:
        if h in wandb.config:
            if h in struct and wandb.config[h] == struct[h]:
                print(f"Parameter '{h}' already had the value {struct[h]}.")
            else:
                print(f"Changing '{h}' from {struct[h]} to {wandb.config[h]}.")
                struct.__setattr__(h, wandb.config[h])
                assert struct[h] == wandb.config[h]


def update_ckt_dir_and_batch_size(config):
    ckt_dir = config.training.lightning_checkpoint["dirpath"]
    if "debug" not in ckt_dir:
        if "accumulate_grad_batches" in wandb.config:
            hyperparam_string = f"seed{wandb.config['seed']}-accumulate{wandb.config['accumulate_grad_batches']}-" \
                                f"lr{wandb.config['learning_rate']}-warmup{wandb.config['warmup_steps']}-name{wandb.config['name']}"
            ckt_dir = ckt_dir.replace('flava-textvision', f'flava-hyperparams-{hyperparam_string}')
            print(f"Hyperparameter sweep detected, changing checkpoint dirpath to {ckt_dir}.")
        else:
            ckt_dir = ckt_dir.replace('flava-textvision', f'flava-textvision-{config.text_perc}t{config.vision_perc}v')
            print(f"Real run, changing checkpoint dirpath to {ckt_dir}.")
        config.training.lightning_checkpoint.__setattr__("dirpath", ckt_dir)

        # This is a hack to make sure we use the memory on the compute node to the fullest.
        if config.model.name == "bert":
            config.training.__setattr__("batch_size", 24)
        elif "flava" in config.model.name:
            if config.training.lightning['precision'] in ["bf16", "16-mixed", "16", 16]:
                config.training.__setattr__("batch_size", 32)
            else:
                config.training.__setattr__("batch_size", 28)
        else:
            raise ValueError(f"Unknown model name {config.model.name}.")
        print(f"Precision is {config.training.lightning['precision']} and batch size is missing, setting batch "
              f"size to {config.training['batch_size']}.")


def assign_huggingface_ram():
    available_memory_gb = get_local_ram()
    huggingface_threshold_gib = 100
    if available_memory_gb > huggingface_threshold_gib:
        torch.set_float32_matmul_precision('medium')
        print(f"Setting float32_matmul_precision to 'medium' to benefit from the Tensor Cores.")

        datasets.config.IN_MEMORY_MAX_SIZE = huggingface_threshold_gib * 1_000_000_000
        print(
            f"Found {available_memory_gb}GBs of RAM available, "
            f"assigning {huggingface_threshold_gib}GBs to HuggingFace datasets (currently {datasets.config.IN_MEMORY_MAX_SIZE} bytes).")
    else:
        print(
            f"Found {available_memory_gb}GBs of RAM available (too little), "
            f"leaving HuggingFace datasets unchanged ({datasets.config.IN_MEMORY_MAX_SIZE} bytes).")


def get_local_ram():
    available_memory_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    available_memory_gb = round(available_memory_bytes / (1024. ** 3), 2)
    return available_memory_gb


def copy_dataset_config_with_training_subset(dataset_info: TrainingSingleDatasetInfo,
                                             percentage: int) -> TrainingSingleDatasetInfo:
    import copy
    dataset_info = copy.deepcopy(dataset_info)
    dataset_info.train[0].split_key_mapping['train'] = f'train[:{percentage}%]'
    return dataset_info
