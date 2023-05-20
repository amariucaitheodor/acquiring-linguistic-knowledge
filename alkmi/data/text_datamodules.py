from typing import Any, Dict, List

import evaluate
import torch
from pytorch_lightning import LightningDataModule
from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
)

from alkmi.data.utils import build_datasets_from_info, collapse_text_columns
from alkmi.definitions import HFDatasetInfo, TEXT_MAX_LENGTH_DEFAULT


def count_words(train, train_infos, validation, validation_infos, after: bool):
    for dataset, split in [(train, train_infos[0].split_key_mapping['train']),
                           (validation, validation_infos[0].split_key_mapping['validation'])]:
        wordcount = evaluate.load("word_count")
        results = wordcount.compute(data=dataset["text"])
        print(f"Split {'after' if after else 'before'} collapsing: {split} ----> "
              f"Total words: {results['total_word_count']}, "
              f"No. of duplicates: {results['total_word_count'] - results['unique_words']}, "
              f"No. of unique: {results['unique_words']}")


class TextDataModule(LightningDataModule):
    def __init__(self,
                 train_infos: List[HFDatasetInfo],
                 val_infos: List[HFDatasetInfo],
                 tokenizer: PreTrainedTokenizerFast,
                 text_columns: List[str],
                 mlm_probability: float,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 **kwargs: Any):
        super().__init__()
        self.name = kwargs['name']
        self.tokenizer = tokenizer

        self.train_dataset_infos = train_infos
        self.val_dataset_infos = val_infos
        if self.val_dataset_infos is None:
            self.val_dataset_infos = train_infos
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.collator = DataCollatorForLanguageModeling(
            self.tokenizer,
            mlm=True,
            mlm_probability=mlm_probability,
            return_tensors="pt"
        )
        self.text_columns = text_columns

    def setup(self, stage=None, should_count_words: bool = False):
        self.train_dataset = build_datasets_from_info(self.train_dataset_infos, split="train")
        self.val_dataset = build_datasets_from_info(self.val_dataset_infos, split="validation")

        if should_count_words:
            count_words(self.train_dataset, self.train_dataset_infos, self.val_dataset, self.val_dataset_infos, False)

        self.train_dataset = collapse_text_columns(self.train_dataset, need_images=False, purpose_msg="MLM training")
        self.val_dataset = collapse_text_columns(self.val_dataset, need_images=False, purpose_msg="MLM validation")

        if should_count_words:
            count_words(self.train_dataset, self.train_dataset_infos, self.val_dataset, self.val_dataset_infos, True)

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
