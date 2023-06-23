from functools import partial
from typing import Any, List, Optional

from pytorch_lightning import LightningDataModule
from transformers import PreTrainedTokenizerFast

from alkmi.data.utils import build_datasets_from_info, process_dataset, count_words
from alkmi.definitions import HFDatasetInfo

from torch.utils.data import DataLoader


class FlavaAblationDataModule(LightningDataModule):
    def __init__(self,
                 train_infos: List[HFDatasetInfo],
                 val_infos: List[HFDatasetInfo],
                 batch_size: int,
                 num_workers: int = 4,
                 **kwargs: Any):
        super().__init__()
        self.name = kwargs['name']

        self.train_dataset_infos = train_infos
        self.val_dataset_infos = val_infos
        if self.val_dataset_infos is None:
            self.val_dataset_infos = train_infos
        self.batch_size = batch_size
        self.num_workers = num_workers

        print(f"{self.name}: batch_size is {batch_size}, num_workers is {num_workers}")

    def setup(self, stage=None):
        self.train_dataset = build_datasets_from_info(self.train_dataset_infos, split="train")
        self.val_dataset = build_datasets_from_info(self.val_dataset_infos, split="validation")

    def train_dataloader(self):
        return self._build_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._build_dataloader(self.val_dataset, shuffle=False)

    def _build_dataloader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=None,
            shuffle=shuffle,
            # uneven batches can cause distributed issues,
            # drop last batch to prevent those.
            # ideally, we don't need to drop these for unimodal cases
            # but just to be safe
            drop_last=True,
        )


class ImageDataModule(FlavaAblationDataModule):
    def setup(self, stage=None):
        super().setup(stage)

        self.train_dataset = self.train_dataset.remove_columns(['text'])
        self.val_dataset = self.val_dataset.remove_columns(['text'])

        self.train_dataset = process_dataset(self.train_dataset, purpose_msg_id="image_training")
        self.val_dataset = process_dataset(self.val_dataset, purpose_msg_id="image_validation")


class MLMDataModule(FlavaAblationDataModule):
    def __init__(
            self,
            train_infos: List[HFDatasetInfo],
            mlm_probability: float,
            tokenizer: PreTrainedTokenizerFast,
            val_infos: Optional[List[HFDatasetInfo]] = None,
            batch_size: int = 32,
            num_workers: int = 4,
            **kwargs: Any,
    ):
        super().__init__(train_infos, val_infos, batch_size, num_workers, **kwargs)

        self.mlm_perc = mlm_probability
        self.tokenizer = tokenizer

    def setup(self, stage=None, should_count_words: bool = False):
        super().setup(stage)

        if should_count_words:
            count_words(self.train_dataset, self.train_dataset_infos, self.val_dataset, self.val_dataset_infos, False)

        self.train_dataset = self.train_dataset.remove_columns(['image'])
        self.val_dataset = self.val_dataset.remove_columns(['image'])

        self.train_dataset = process_dataset(self.train_dataset, purpose_msg_id="text_training",
                                             mlm_perc=self.mlm_perc, tokenizer=self.tokenizer)
        self.val_dataset = process_dataset(self.val_dataset, purpose_msg_id="text_validation",
                                           mlm_perc=self.mlm_perc, tokenizer=self.tokenizer)


class VLDataModule(FlavaAblationDataModule):
    def __init__(
            self,
            train_infos: List[HFDatasetInfo],
            val_infos: List[HFDatasetInfo],
            mlm_probability: float,
            batch_size: int,
            num_workers: int = 4,
            itm_probability: float = 0.1,
            # text_columns implicitly only 'text' here!
            **kwargs,
    ):
        super().__init__(train_infos, val_infos, batch_size, num_workers, **kwargs)

        self.mlm_perc = mlm_probability
        self.itm_probability = itm_probability

    def setup(self, stage=None):
        super().setup(stage)

        vl_transform = lambda dataset: partial(
            ITMTransform(),
            dataset=dataset.filter(
                lambda examples: [True] * len(examples["text"]),  # or "image" ...
                batched=True,
                num_proc=32,
                batch_size=100,
                desc="Creating a copy of the dataset (to be ITM-transformed)..."
            ),
            itm_probability=self.itm_probability,
        )

        self.train_dataset = process_dataset(self.train_dataset, purpose_msg_id="vl_training",
                                             itm_probability=self.itm_probability, mlm_perc=self.mlm_perc)
        self.val_dataset = process_dataset(self.val_dataset, purpose_msg_id="vl_validation",
                                           itm_probability=self.itm_probability, mlm_perc=self.mlm_perc)

        # https://discuss.huggingface.co/t/use-existing-dataset-with-a-generator/36219/4
