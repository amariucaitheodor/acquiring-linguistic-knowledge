from typing import List

from datasets import concatenate_datasets, load_dataset, Image
from datasets.utils.file_utils import get_datasets_user_agent

from alkmi.definitions import HFDatasetInfo

DATASETS_USER_AGENT = get_datasets_user_agent()


def build_datasets_from_info(dataset_infos: List[HFDatasetInfo], split: str = "train"):
    dataset_list = []
    for dataset_info in dataset_infos:
        current_dataset = load_dataset(
            dataset_info.key,
            dataset_info.subset,
            split=dataset_info.split_key_mapping[split],
            use_auth_token=True,
            num_proc=32,
            **dataset_info.extra_kwargs,
        )

        if dataset_info.remove_columns is not None:
            current_dataset = current_dataset.remove_columns(dataset_info.remove_columns)

        if dataset_info.rename_columns is not None:
            for rename in dataset_info.rename_columns:
                current_dataset = current_dataset.rename_column(rename[0], rename[1])

        dataset_list.append(current_dataset)

    return concatenate_datasets(dataset_list)


WIT_ALT_TEXT_COLUMNS = ["context_page_description", "context_section_description",
                        "caption_alt_text_description", "caption_attribution_description"]


def collapse_wit_text(batch):
    # WARNING: This increases the disk space requirement by a factor of ~4
    original_len = len(batch["text"])
    for i in range(original_len):
        for field in WIT_ALT_TEXT_COLUMNS:
            if batch[field][i] is not None:
                if "image" in batch: batch["image"].append(batch["image"][i])
                batch["text"].append(
                    batch[field][i].split("English: ")[1] if batch[field][i].startswith("English: ")
                    else batch[field][i])
    return batch


def collapse_text_columns(dataset, need_images: bool, purpose_msg: str, num_proc: int = 16, batch_size: int = 100):
    if len(dataset.column_names) > 1:
        if 'image' in dataset.column_names:
            dataset = dataset.cast_column("image", Image(decode=False))  # MUCH faster processing
        dataset = dataset.map(
            collapse_wit_text,
            batched=True,
            num_proc=num_proc,
            batch_size=batch_size,
            remove_columns=WIT_ALT_TEXT_COLUMNS,
            load_from_cache_file=True,  # MUCH faster processing
            desc=f"Collapsing WiT text for {purpose_msg}",
        )
        if 'image' in dataset.column_names and need_images:
            dataset = dataset.cast_column("image", Image(decode=True))
    return dataset
