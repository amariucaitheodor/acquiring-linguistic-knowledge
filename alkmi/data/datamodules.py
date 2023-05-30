from functools import partial
from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning import LightningDataModule
from transformers import (
    DataCollatorForLanguageModeling,
    FlavaProcessor, PreTrainedTokenizerFast,
)

from alkmi.data import utils
from alkmi.data.transforms import ITMTransform
from alkmi.data.utils import build_datasets_from_info, collapse_text_columns, count_words
from alkmi.definitions import HFDatasetInfo, TEXT_MAX_LENGTH_DEFAULT, VL_MAX_LENGTH_DEFAULT


class FlavaAblationDataModule(LightningDataModule):
    def __init__(self,
                 train_infos: List[HFDatasetInfo],
                 val_infos: List[HFDatasetInfo],
                 batch_size: int,
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

        print(f"{self.name}: batch_size is {batch_size}, num_workers is {num_workers}")

    def setup(self, stage=None):
        self.train_dataset = build_datasets_from_info(self.train_dataset_infos, split="train")
        self.val_dataset = build_datasets_from_info(self.val_dataset_infos, split="validation")

    def train_dataloader(self):
        return self._build_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._build_dataloader(self.val_dataset, shuffle=False)

    def _build_dataloader(self, dataset, shuffle: bool):
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


class ImageDataModule(FlavaAblationDataModule):
    def setup(self, stage=None):
        super().setup(stage)

        self.train_dataset = self.train_dataset.remove_columns(utils.WIT_OTHER_TEXT_COLUMNS + ['text'])
        self.val_dataset = self.val_dataset.remove_columns(utils.WIT_OTHER_TEXT_COLUMNS + ['text'])


class MLMDataModule(FlavaAblationDataModule):
    def __init__(
            self,
            text_columns: List[str],
            train_infos: List[HFDatasetInfo],
            mlm_probability: float,
            tokenizer: PreTrainedTokenizerFast,
            val_infos: Optional[List[HFDatasetInfo]] = None,
            batch_size: int = 32,
            num_workers: int = 4,
            **kwargs: Any,
    ):
        super().__init__(train_infos, val_infos, batch_size, num_workers, **kwargs)

        if tokenizer is None:
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = tokenizer

        self.collator = DataCollatorForLanguageModeling(
            self.tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
            return_tensors="pt"
        )
        self.text_columns = text_columns

    def setup(self, stage=None, should_count_words: bool = False):
        super().setup(stage)

        if should_count_words:
            count_words(self.train_dataset, self.train_dataset_infos, self.val_dataset, self.val_dataset_infos, False)

        self.train_dataset = collapse_text_columns(self.train_dataset, need_images=False, purpose_msg="MLM training")
        self.val_dataset = collapse_text_columns(self.val_dataset, need_images=False, purpose_msg="MLM validation")

        if should_count_words:
            count_words(self.train_dataset, self.train_dataset_infos, self.val_dataset, self.val_dataset_infos, True)

    def _build_collator(self, inputs: List[Dict[str, Any]]):
        text_to_process = []
        for col in self.text_columns:
            for i in inputs:
                text_to_process.append(i[col])
        batch = self.tokenizer(
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
            mlm_probability: float,
            batch_size: int,
            num_workers: int = 4,
            itm_probability: float = 0.1,
            # text_columns implicitly only 'text' here!
            **kwargs,
    ):
        super().__init__(train_infos, val_infos, batch_size, num_workers, **kwargs)
        self.collator = DataCollatorForLanguageModeling(
            self.processor.tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
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
                desc="Creating a copy of the dataset (to be ITM-transformed)..."
            ),
            itm_probability=self.itm_probability,
        )

        self.train_dataset = collapse_text_columns(self.train_dataset, need_images=True, purpose_msg="VL training")
        self.train_dataset.set_transform(vl_transform(self.train_dataset))

        self.val_dataset = collapse_text_columns(self.val_dataset, need_images=True, purpose_msg="VL validation")
        self.val_dataset.set_transform(vl_transform(self.val_dataset))

        # https://discuss.huggingface.co/t/use-existing-dataset-with-a-generator/36219/4

    def _build_collator(self, inputs):
        batch = super()._build_collator(inputs)
        # https://github.com/huggingface/transformers/issues/22103
        batch["input_ids_masked"], batch["mlm_labels"] = self.collator.torch_mask_tokens(
            inputs=batch["input_ids"].detach().clone(),
            special_tokens_mask=batch.pop("special_tokens_mask", None)
        )
        batch["itm_labels"] = torch.tensor([i["itm_labels"] for i in inputs])
        return batch
