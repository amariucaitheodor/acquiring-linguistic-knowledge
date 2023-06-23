import io
import random
import warnings
from functools import partial
from typing import List, Optional

import evaluate
import torch
from datasets import concatenate_datasets, load_dataset, Image, Dataset, DownloadMode
from datasets.utils.file_utils import get_datasets_user_agent
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast

from alkmi.definitions import HFDatasetInfo, VL_MAX_LENGTH_DEFAULT, TEXT_MAX_LENGTH_DEFAULT
from models.flava import FlavaProcessor

DATASETS_USER_AGENT = get_datasets_user_agent()


def count_words(train, train_infos, validation, validation_infos, after: bool):
    for dataset, split in [(train, train_infos[0].split_key_mapping['train']),
                           (validation, validation_infos[0].split_key_mapping['validation'])]:
        wordcount = evaluate.load("word_count")
        results = wordcount.compute(data=dataset["text"])
        print(f"Split {'after' if after else 'before'} collapsing: {split} ----> "
              f"Total words: {results['total_word_count']}, "
              f"No. of duplicates: {results['total_word_count'] - results['unique_words']}, "
              f"No. of unique: {results['unique_words']}")


def build_datasets_from_info(dataset_infos: List[HFDatasetInfo], split: str = "train"):
    dataset_list = []
    for dataset_info in dataset_infos:
        current_dataset = load_dataset(
            dataset_info.key,
            dataset_info.subset,
            split=dataset_info.split_key_mapping[split],
            use_auth_token=True,
            num_proc=32,
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
            save_infos=True,
            **dataset_info.extra_kwargs,
        )

        if dataset_info.remove_columns is not None:
            current_dataset = current_dataset.remove_columns(dataset_info.remove_columns)

        if dataset_info.rename_columns is not None:
            for rename in dataset_info.rename_columns:
                current_dataset = current_dataset.rename_column(rename[0], rename[1])

        dataset_list.append(current_dataset)

    return concatenate_datasets(dataset_list)


# "caption_attribution_description" is not used because of its multilingual nature
WIT_OTHER_TEXT_COLUMNS = ["context_page_description", "context_section_description", "caption_alt_text_description"]


def process_batch(batch: dict,
                  itm_probability: Optional[float],
                  processor: Optional[FlavaProcessor],
                  tokenizer: Optional[PreTrainedTokenizerFast],
                  collator: Optional[DataCollatorForLanguageModeling]):
    if "text" in batch:
        warnings.warn("Collapsing additional text fields, disk space requirements will increase by a factor of ~4")
        original_len = len(batch["text"])
        for i in range(original_len):
            for field in WIT_OTHER_TEXT_COLUMNS:
                if batch[field][i] is not None:
                    if "image" in batch:
                        batch["image"].append(batch["image"][i])
                    batch["text"].append(batch[field][i])

    if itm_probability:
        length = len(batch["text"])
        itm_transformed_batch = {"itm_labels": torch.full(size=[length], fill_value=1.0 - itm_probability)}
        itm_transformed_batch["itm_labels"] = torch.bernoulli(input=itm_transformed_batch["itm_labels"]).long()
        for i in range(len(itm_transformed_batch["itm_labels"])):
            if itm_transformed_batch["itm_labels"][i] == 0:
                original = batch["text"][i]
                while batch["text"][i] == original:  # rejection sampling
                    batch["text"][i] = batch["text"][random.randint(0, length - 1)]
        # (ITM) Original image order is kept, text could change:
        itm_transformed_batch.update({"image": batch["image"]})
        itm_transformed_batch.update({"text": batch["text"]})
        batch = itm_transformed_batch

    if processor:
        images = None
        if 'image' in batch:
            if type(batch["image"][0]) == dict:
                images = [Image.open(io.BytesIO(img['image']['bytes'])) for img in batch["image"]]
            else:
                images = batch["image"]

        batch = processor(
            text=batch["text"] if "text" in batch else None,
            images=images,
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
    elif tokenizer:
        batch = tokenizer(
            text=batch["text"],
            return_tensors="pt",
            padding="max_length",
            add_special_tokens=True,
            truncation=True,  # very important!
            max_length=TEXT_MAX_LENGTH_DEFAULT,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_special_tokens_mask=True,
        )

    if collator:
        # https://github.com/huggingface/transformers/issues/22103
        batch["input_ids_masked"], batch["mlm_labels"] = collator.torch_mask_tokens(
            inputs=batch["input_ids"].detach().clone(),
            special_tokens_mask=batch.pop("special_tokens_mask", None)
        )

    return batch


def process_dataset(dataset: Dataset,
                    purpose_msg_id: str,
                    mlm_perc: Optional[float] = None,
                    itm_probability: Optional[float] = None,
                    tokenizer: Optional[PreTrainedTokenizerFast] = None,
                    num_proc: int = 1,
                    batch_size: int = 100
                    ):
    if purpose_msg_id == 'ppl_evaluation':
        processor = None
        assert not mlm_perc, "MLM percentage should not be provided for ppl dataset processing"
        assert not tokenizer, "Tokenizer should not be provided for ppl dataset processing"
        assert not itm_probability, "ITM probability should not be provided for ppl dataset processing"
        collator = None
    elif purpose_msg_id.startswith('image'):
        processor = FlavaProcessor.from_pretrained("facebook/flava-full")
        assert not mlm_perc, "MLM percentage should not be provided for image dataset processing"
        assert not tokenizer, "Tokenizer should not be provided for image dataset processing"
        assert not itm_probability, "ITM probability should not be provided for image dataset processing"
        collator = None
    elif purpose_msg_id.startswith('text'):
        processor = FlavaProcessor.from_pretrained("facebook/flava-full") if not tokenizer else None
        assert mlm_perc, "MLM percentage should be provided for text dataset processing"
        assert not itm_probability, "ITM probability should not be provided for text dataset processing"
        collator = DataCollatorForLanguageModeling(tokenizer if tokenizer else processor.tokenizer,
                                                   mlm=True, mlm_probability=mlm_perc, return_tensors="pt")
    elif purpose_msg_id.startswith('vl'):
        processor = FlavaProcessor.from_pretrained("facebook/flava-full")
        assert mlm_perc, "MLM percentage should be provided for vl dataset processing"
        assert not tokenizer, "Tokenizer should not be provided for vl dataset processing"
        assert itm_probability, "ITM probability should be provided for vl dataset processing"
        collator = DataCollatorForLanguageModeling(processor.tokenizer, mlm=True,
                                                   mlm_probability=mlm_perc, return_tensors="pt")
    else:
        raise ValueError(f"Unknown purpose message id: {purpose_msg_id}")

    dataset = dataset.map(
        partial(process_batch,
                processor=processor,
                tokenizer=tokenizer,
                collator=collator,
                itm_probability=itm_probability),
        batched=True,
        num_proc=num_proc,
        batch_size=batch_size,
        remove_columns=WIT_OTHER_TEXT_COLUMNS + ["image_url"] +
                       (["image"] if "image" in dataset.column_names else []) +
                       (["text"] if "text" in dataset.column_names and not purpose_msg_id.startswith('ppl') else []),
        load_from_cache_file=False,  # MUCH faster processing
        cache_file_name=purpose_msg_id,
        new_fingerprint=purpose_msg_id,
        desc=f"Processing WiT dataset for {purpose_msg_id}",
    )

    print(f"Finished processing WiT dataset for {purpose_msg_id}. "
          f"A sample looks like: {[(k, type(v)) for k, v in dataset[0].items()]}")
    return dataset
