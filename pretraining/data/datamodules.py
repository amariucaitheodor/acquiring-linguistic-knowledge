from functools import partial
from typing import Any, Dict, List, Optional

import torch
from datasets import Image
from pytorch_lightning import LightningDataModule
from transformers import (
    DataCollatorForLanguageModeling,
    FlavaProcessor,
)

from pretraining.data import utils
from pretraining.data.transforms import ITMTransform
from pretraining.data.utils import build_datasets_from_info
from pretraining.definitions import HFDatasetInfo, TEXT_MAX_LENGTH_DEFAULT, VL_MAX_LENGTH_DEFAULT


class FlavaAblationDataModule(LightningDataModule):
    def __init__(self,
                 train_infos: List[HFDatasetInfo],
                 val_infos: List[HFDatasetInfo],
                 batch_size: int = 32,
                 num_workers: int = 4,
                 **kwargs: Any):
        super().__init__()
        self.name = kwargs['name']
        self.processor = FlavaProcessor.from_pretrained("facebook/flava-full")

        self.train_dataset_infos = train_infos
        self.val_dataset_infos = val_infos
        if self.val_dataset_infos is None:
            self.val_dataset_infos = train_infos
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = build_datasets_from_info(self.train_dataset_infos, split="train")
        self.val_dataset = build_datasets_from_info(self.val_dataset_infos, split="validation")

    def train_dataloader(self):
        return self._build_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._build_dataloader(self.val_dataset, shuffle=False)

    def _build_dataloader(self, dataset, shuffle=True):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=None,
            shuffle=shuffle,
            collate_fn=self._build_collator,
            # uneven batches can cause distributed issues,
            # drop last batch to prevent those.
            # ideally, we don't need to drop these for unimodal cases
            # but just to be safe
            drop_last=True,
        )

    def _build_collator(self, inputs: List[Dict[str, Any]]):
        return self.processor(
            text=[i['text'] for i in inputs] if 'text' in inputs[0] else None,
            images=[i['image'].convert('RGB') for i in inputs] if 'image' in inputs[0] else None,
            return_tensors="pt",
            padding="max_length",
            add_special_tokens=True,
            truncation=True,  # very important!
            max_length=VL_MAX_LENGTH_DEFAULT,
            return_image_mask=True,
            return_codebook_pixels=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_special_tokens_mask=True,
        )


class ImagenetEvalDataModule(FlavaAblationDataModule):

    def test_dataloader(self):
        return self.val_dataloader()

    def _build_collator(self, inputs: List[Dict[str, Any]]):
        # for imagenet eval we don't need to use the processor
        return inputs


class ImageDataModule(FlavaAblationDataModule):
    def setup(self, stage=None):
        super().setup(stage)

        self.train_dataset = self.train_dataset.remove_columns(utils.WIT_ALT_TEXT_COLUMNS + ['text'])
        self.val_dataset = self.val_dataset.remove_columns(utils.WIT_ALT_TEXT_COLUMNS + ['text'])


class MLMDataModule(FlavaAblationDataModule):
    def __init__(
            self,
            text_columns: List[str],
            train_infos: List[HFDatasetInfo],
            val_infos: Optional[List[HFDatasetInfo]] = None,
            mlm_probability: float = 0.15,
            batch_size: int = 32,
            num_workers: int = 4,
            **kwargs: Any,
    ):
        super().__init__(train_infos, val_infos, batch_size, num_workers, **kwargs)
        self.collator = DataCollatorForLanguageModeling(
            self.processor.tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
            return_tensors="pt"
        )
        self.text_columns = text_columns

    def setup(self, stage=None):
        super().setup(stage)

        self.train_dataset = self.train_dataset.remove_columns('image')
        self.train_dataset = self.train_dataset.map(
            utils.collapse_wit_text,
            batched=True,
            num_proc=32,
            batch_size=100,
            remove_columns=utils.WIT_ALT_TEXT_COLUMNS
        )
        self.val_dataset = self.val_dataset.remove_columns('image')
        self.val_dataset = self.val_dataset.map(
            utils.collapse_wit_text,
            batched=True,
            num_proc=32,
            batch_size=100,
            remove_columns=utils.WIT_ALT_TEXT_COLUMNS
        )

    def _build_collator(self, inputs: List[Dict[str, Any]]):
        text_to_process = []
        for col in self.text_columns:
            for i in inputs:
                text_to_process.append(i[col])
        batch = self.processor.tokenizer(
            text=text_to_process,
            return_tensors="pt",
            padding="max_length",
            add_special_tokens=True,
            truncation=True,  # very important!
            max_length=TEXT_MAX_LENGTH_DEFAULT,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_special_tokens_mask=True,
        )
        # https://github.com/huggingface/transformers/issues/22103
        batch["input_ids_masked"], batch["mlm_labels"] = self.collator.torch_mask_tokens(
            inputs=batch["input_ids"].detach().clone(),
            special_tokens_mask=batch.pop("special_tokens_mask", None)
        )
        return batch


class VLDataModule(FlavaAblationDataModule):
    def __init__(
            self,
            train_infos: List[HFDatasetInfo],
            val_infos: List[HFDatasetInfo],
            mlm_probablity: float = 0.15,
            batch_size: int = 32,
            num_workers: int = 4,
            itm_probability: float = 0.1,
            # text_columns implicitly only 'text' here!
            **kwargs,
    ):
        super().__init__(train_infos, val_infos, batch_size, num_workers, **kwargs)
        self.collator = DataCollatorForLanguageModeling(
            self.processor.tokenizer,
            mlm=True,
            mlm_probability=mlm_probablity,
            return_tensors="pt"
        )
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
            ),  # Pass a copy (to transform), this is the purpose of this filtering action
            itm_probability=self.itm_probability,
        )

        self.train_dataset = self.train_dataset.cast_column("image", Image(decode=False))  # MUCH faster processing
        self.train_dataset = self.train_dataset.map(
            utils.collapse_wit_text,
            batched=True,
            num_proc=32,
            batch_size=100,
            remove_columns=utils.WIT_ALT_TEXT_COLUMNS,
            load_from_cache_file=True  # MUCH faster processing
        )
        self.train_dataset.set_transform(vl_transform(self.train_dataset))
        self.train_dataset = self.train_dataset.cast_column("image", Image(decode=True))  # MUCH faster processing

        self.val_dataset = self.val_dataset.cast_column("image", Image(decode=False))  # MUCH faster processing
        self.val_dataset = self.val_dataset.map(
            utils.collapse_wit_text,
            batched=True,
            num_proc=32,
            batch_size=100,
            remove_columns=utils.WIT_ALT_TEXT_COLUMNS,
            load_from_cache_file=True  # MUCH faster processing
        )
        self.val_dataset.set_transform(vl_transform(self.val_dataset))
        self.val_dataset = self.val_dataset.cast_column("image", Image(decode=True))  # MUCH faster processing

        # https://discuss.huggingface.co/t/use-existing-dataset-with-a-generator/36219/4

    def _build_collator(self, inputs):
        batch = super()._build_collator(inputs)
        # https://github.com/huggingface/transformers/issues/22103
        batch["input_ids_masked"], batch["mlm_labels"] = self.collator.torch_mask_tokens(
            inputs=batch["input_ids"].detach().clone(),
            special_tokens_mask=batch.pop("special_tokens_mask", None)
        )
        return batch
