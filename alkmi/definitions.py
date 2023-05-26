# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import MISSING

TEXT_MAX_LENGTH_DEFAULT = 512
VL_MAX_LENGTH_DEFAULT = 77


def _default_split_key_mapping():
    return {x: x for x in ["train", "validation", "test"]}


@dataclass
class DatasetInfo:
    key: str = MISSING


@dataclass
class HFDatasetInfo(DatasetInfo):
    key: str = MISSING
    subset: Optional[str] = None
    remove_columns: Optional[List[str]] = None
    # Any is actually list of pairs for renaming the column A to B
    # limited to Any because of OmegaConf limitations
    rename_columns: Optional[List[Any]] = None
    # TODO: Look if we can add text column option and encode transform settings here.
    split_key_mapping: Optional[Dict[str, str]] = field(
        default_factory=_default_split_key_mapping
    )
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingSingleDatasetInfo:
    train: List[DatasetInfo] = field(default_factory=lambda: [HFDatasetInfo()])
    val: Optional[List[HFDatasetInfo]] = None
    batch_size: Optional[int] = None
    num_workers: Optional[int] = None
    allow_uneven_batches: bool = False
    datamodule_extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingDatasetsInfo:
    selected: List[str] = field(default_factory=lambda: ["ablation"])
    sampling_temperature: float = 1.0  # sampling proportional to the dataset sizes (by default)
    ablation: Optional[TrainingSingleDatasetInfo] = None
    num_classes: int = MISSING


@dataclass
class TrainingArguments:
    # Any lightning args to be pushed here
    lightning: Dict[str, Any] = field(default=dict)
    lightning_checkpoint: Optional[Dict[str, Any]] = None
    lightning_load_from_checkpoint: Optional[str] = None
    seed: int = -1
    batch_size: int = 8
    num_workers: int = 2
    learning_rate: float = 0.003
    learning_rate_text_submodel: float = 0.0005
    adam_eps: float = 1e-08
    adam_weight_decay: float = 0.01
    adam_betas: Tuple[float, float] = field(default_factory=lambda: (0.9, 0.999))
    warmup_steps: int = 2000
    use_wandb: bool = False


@dataclass
class ModelArguments:
    pretrained: Optional[str] = None
    mlm_perc: float = 0.15
    name: str = 'flava'


@dataclass
class AblationArguments:
    text_perc: int = 100
    vision_perc: int = 100
    datasets: TrainingDatasetsInfo = TrainingDatasetsInfo()
    training: TrainingArguments = TrainingArguments()
    model: ModelArguments = ModelArguments()
