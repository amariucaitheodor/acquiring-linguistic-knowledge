from typing import List

import evaluate
from datasets import concatenate_datasets, load_dataset, Image, Dataset, DownloadMode
from datasets.utils.file_utils import get_datasets_user_agent

from alkmi.definitions import HFDatasetInfo

DATASETS_USER_AGENT = get_datasets_user_agent()


def count_words(train: Dataset, validation: Dataset, after: bool):
    for dataset in [train, validation]:
        print(f"Counting words for split {dataset.split} {'after' if after else 'before (i.e. captions only)'} "
              f"collapsing...")
        wordcount = evaluate.load("word_count")
        results = wordcount.compute(data=dataset["text"])
        print(f"Split {'after' if after else 'before (i.e. captions only)'} collapsing: {dataset.split} ----> "
              f"Total words: {results['total_word_count']}, "
              f"No. of unique: {results['unique_words']}")


def build_datasets_from_info(dataset_infos: List[HFDatasetInfo], split: str) -> Dataset:
    assert len(dataset_infos) == 1, "Only one dataset supported for now"

    dataset_info = dataset_infos[0]
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

    return current_dataset


# "caption_attribution_description" is not used because of its multilingual nature
WIT_OTHER_TEXT_COLUMNS = ["context_page_description", "context_section_description", "caption_alt_text_description"]


def collapse_wit_text(batch):
    # WARNING: This increases the disk space requirement by a factor of ~4
    original_len = len(batch["text"])
    for i in range(original_len):
        for field in WIT_OTHER_TEXT_COLUMNS:
            if batch[field][i] is not None:
                if "image" in batch:
                    batch["image"].append(batch["image"][i])
                batch["text"].append(batch[field][i])
    return batch


def collapse_text_columns(dataset: Dataset, need_images: bool, num_proc: int = 1, batch_size: int = 200):
    if len(dataset.column_names) > 1:
        if 'image' in dataset.column_names:
            dataset = dataset.cast_column("image", Image(decode=False))  # MUCH faster processing
        dataset = dataset.map(
            collapse_wit_text,
            batched=True,
            num_proc=num_proc,
            batch_size=batch_size,
            remove_columns=WIT_OTHER_TEXT_COLUMNS + ["image_url"],
            load_from_cache_file=True,  # MUCH faster processing
            cache_file_name=dataset.split.__str__().replace(':', ''),
            new_fingerprint=dataset.split.__str__().replace(':', ''),
            writer_batch_size=3000,
            desc=f"Collapsing WiT text for split {dataset.split} of {dataset.info.builder_name}",
        )
        if 'image' in dataset.column_names:
            if need_images:
                dataset = dataset.cast_column("image", Image(decode=True))
            else:
                dataset = dataset.remove_columns("image")
    return dataset
