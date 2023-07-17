import os
import time
from typing import List

import datasets
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf
from transformers import BertTokenizerFast, RobertaTokenizerFast

from data.datamodules import MLMDataModule, ImageDataModule, VLDataModule
from data.multidata import MultiDataModule
from alkmi.definitions import TrainingSingleDatasetInfo, TrainingArguments, AblationArguments, ModelArguments


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


def build_model_kwargs(
        training_config: TrainingArguments, model_config: ModelArguments
):
    return {
        "pretrained": model_config.pretrained,
        "half_size": model_config.half_size,
        "precision": training_config.precision,  # needed for deciding whether to use parameter sets
        "learning_rate": training_config.learning_rate,
        "learning_rate_text_submodel": training_config.learning_rate_text_submodel,
        "adam_eps": training_config.adam_eps,
        "adam_betas": training_config.adam_betas,
        "warmup_steps": training_config.warmup_steps,
        "adam_weight_decay": training_config.adam_weight_decay,
        "max_steps": 450000,  # fixed for now...
    }


def build_config():
    cli_conf = OmegaConf.from_cli()
    if "config" not in cli_conf:
        raise ValueError(
            "Please pass 'config' to specify configuration yaml file for running FLAVA"
        )
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = instantiate(yaml_conf)
    cli_conf.pop("config")
    config: AblationArguments = OmegaConf.merge(conf, cli_conf)

    assert 0 <= config.text_perc <= 100, "'text_perc' ablation study percentage must be between 0 and 100"
    assert 0 <= config.vision_perc <= 100, "'vision_perc' ablation study percentage must be between 0 and 100"

    if config.model.name in ['bert', 'roberta', 'deberta']:
        assert config.vision_perc == 0, f"{config.model.name} can only be trained on text (vision_perc was {config.vision_perc})"

    return config


def initialize_multidatamodule(config: AblationArguments) -> MultiDataModule:
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
            mlm_probability=config.model.mlm_perc,  # https://arxiv.org/abs/2202.08005
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

        tokenizer = None
        if config.model.name == 'bert':
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        elif config.model.name == 'roberta':
            tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

        mlm_datamodule = MLMDataModule(
            **build_datamodule_kwargs(text_config, config.training),
            tokenizer=tokenizer,
            mlm_probability=config.model.mlm_perc,  # https://arxiv.org/abs/2202.08005
            name="MLMDataModule"
        )
        modules.append(mlm_datamodule)

    if len(modules) == 3:
        sampling_weights = [multimodal_perc, config.vision_perc, config.text_perc]
        print(f"Dataset sizes (un-normalized): {sampling_weights}")
        sampling_weights = [float(w) / sum(sampling_weights) for w in sampling_weights]
        print(f"Dataset sizes (normalized): {sampling_weights}")
        sampling_weights = [w ** config.datasets.sampling_temperature for w in sampling_weights]
        print(f"Sampling weights after temperature ({config.datasets.sampling_temperature}): {sampling_weights}")
    else:
        assert len(modules) == 1
        sampling_weights = [1.]
        print(f"Only one modality, sampling weight will be {sampling_weights}")

    # Table A.2 in https://arxiv.org/pdf/2112.04482.pdf
    datamodule = MultiDataModule(datamodules=modules, sampling_weights=sampling_weights)
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


def update_ckeckpoint_dir(config, batch_size: int):
    ckt_dir = config.training.lightning_checkpoint["dirpath"]
    if "debug" not in ckt_dir:
        if len(wandb.config.items()) > 1:
            print("[update_ckeckpoint_dir] Detected hyperparameter run!")
            hparam_string = "-".join([f"{hparam}({value})" for hparam, value in wandb.config.items()])
            ckt_dir += f'{time.strftime("date(%Y-%m-%d)_time(%H:%M:%S)")}/{hparam_string[:-1]}/'
        else:
            ckt_dir += f'text{config.text_perc}-vision{config.vision_perc}/' \
                       f'bs{batch_size}_seed{config.training.seed}_{config.training.precision}/'
        print(f"[update_ckeckpoint_dir] Setting checkpoint dirpath to {ckt_dir}")
        config.training.lightning_checkpoint.__setattr__("dirpath", ckt_dir)


def assign_huggingface_ram():
    available_memory_gb = get_local_ram()
    huggingface_threshold_gib = 5
    if available_memory_gb > huggingface_threshold_gib:
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
