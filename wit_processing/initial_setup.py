import io
import json
import urllib
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import PIL.Image
import datasets
from datasets.utils.file_utils import get_datasets_user_agent

from alkmi.data.utils import WIT_ALT_TEXT_COLUMNS

USER_AGENT = get_datasets_user_agent()


def fetch_single_image(image_url, timeout, retries):
    for _ in range(retries + 1):
        try:
            # "Wikimedia commons" down-sampling URL technique...
            # https://stackoverflow.com/questions/72544973/how-to-get-a-lower-resolution-version-of-an-image-from-wikimedia
            size = 512
            url = 'upload.wikimedia.org/wikipedia/commons/'
            for prefix in [f'https://{url}', f'http://{url}']:
                if image_url.startswith(prefix):
                    suffix = image_url.split(prefix)[1]
                    file_name = image_url.split('/')[-1]
                    image_url = f'{prefix}thumb/{suffix}/{size}px-{file_name}.png'
                    break
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
            break
        except Exception:
            image = None
    return image


def fetch_images(batch, num_threads, timeout=None, retries=3):
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(fetch_single_image_with_args, batch["image_url"]))
    return batch


def process_text(batch, num_threads: int, collapse: bool):
    # This extracts the alternative text fields from the "meta" field
    with ThreadPoolExecutor(max_workers=num_threads) as executor:  # "text" is the caption
        for field in WIT_ALT_TEXT_COLUMNS:
            batch[field] = list(executor.map(lambda meta: json.loads(meta)[field], batch["meta"]))
    if collapse:  # This pairs every image with each corresponding alternative text field
        # WARNING: This increases the disk space requirement by a factor of ~4
        original_len = len(batch["text"])
        for i in range(original_len):
            for field in WIT_ALT_TEXT_COLUMNS:
                if batch[field][i] is not None:
                    batch["image"].append(batch["image"][i])
                    batch["text"].append(
                        batch[field][i].split("English: ")[1] if batch[field][i].startswith("English: ")
                        else batch[field][i])
        for field in WIT_ALT_TEXT_COLUMNS: del batch[field]
    return batch


text_and_image_both_present = lambda examples: [text is not None and image is not None for text, image in
                                                zip(examples["text"], examples["image"])]

if __name__ == "__main__":
    NUM_THREADS = 64
    MINI_VERSION = False
    COLLAPSE_TEXT = False

    if MINI_VERSION:
        dataset = datasets.load_dataset("facebook/pmd", "wit", split='train', use_auth_token=True, num_proc=48) \
            .select(list(range(1000)))
        dataset = dataset.map(fetch_images,
                              fn_kwargs={"num_threads": NUM_THREADS},
                              batched=True, batch_size=50,
                              num_proc=16, remove_columns=["image_url"])
        dataset = dataset.filter(
            text_and_image_both_present,
            batched=True,
            num_proc=6,
            batch_size=25,
            desc=f"Filtering out data with missing content from wit_mini"
        )
        dataset = dataset.train_test_split(test_size=0.1, train_size=0.9)  # First split, then collapse text
        for split in ['train', 'test']:
            print("Processing split", split)
            dataset[split] = dataset[split].map(process_text, batched=True, num_proc=24, batch_size=100,
                                                fn_kwargs={"num_threads": NUM_THREADS, "collapse": COLLAPSE_TEXT},
                                                remove_columns=["meta", "source"])
        dataset.push_to_hub(f"theodor1289/{'wit_collapsed' if COLLAPSE_TEXT else 'wit'}", max_shard_size="500MB",
                            private=False)
    else:
        STAGE = 1
        SAVE_DISK_SHARD_SIZE, UPLOAD_SHARD_SIZE, SAVE_NUM_PROC = "10GB", "500MB", 32
        SCRATCH_PATH, FINAL_PATH = '/cluster/scratch/tamariucai', '/cluster/work/cotterell/tamariucai'

        if STAGE in [1, 0]:
            print(f"Running STEP 1: get WiT, process the images and save to disk")
            dataset = datasets.load_dataset("facebook/pmd", "wit", split='train', use_auth_token=True, num_proc=12)
            dataset = dataset.map(fetch_images,
                                  fn_kwargs={"num_threads": NUM_THREADS},
                                  batched=True,
                                  batch_size=10,
                                  num_proc=16,
                                  remove_columns=["image_url"])
            dataset.save_to_disk(f'{SCRATCH_PATH}/wit_images/', max_shard_size=SAVE_DISK_SHARD_SIZE,
                                 num_proc=SAVE_NUM_PROC)
            print(f"dataset.cleanup_cache_files(): {dataset.cleanup_cache_files()}")

        if STAGE in [2, 0]:
            print(f"Running STEP 2: load from disk, filter missing content, save to disk")
            dataset = datasets.load_from_disk(f'{SCRATCH_PATH}/wit_images/')
            dataset = dataset.filter(
                text_and_image_both_present,
                batched=True,
                num_proc=6,
                batch_size=25,
                desc=f"Filtering out data with missing content from wit_images"
            )
            dataset.save_to_disk(f'{SCRATCH_PATH}/wit_filtered/', max_shard_size=SAVE_DISK_SHARD_SIZE,
                                 num_proc=SAVE_NUM_PROC)

        if STAGE in [3, 0]:
            print(f"Running STEP 3: load filtered WiT; split, process the text, then save to disk")
            dataset = datasets.load_from_disk(f'{SCRATCH_PATH}/wit_filtered/')
            dataset = dataset.train_test_split(test_size=0.1, train_size=0.9)

            print(f"Midpoint STEP 3: processing the text AFTER splitting the dataset (collapse={COLLAPSE_TEXT})...")
            for split in ['train', 'test']:
                print("Processing split", split)
                dataset[split] = dataset[split].map(process_text, batched=True, num_proc=24, batch_size=100,
                                                    fn_kwargs={"num_threads": NUM_THREADS, "collapse": COLLAPSE_TEXT},
                                                    remove_columns=["meta", "source"])

            print(f'Final STEP 3: saving to disk at {FINAL_PATH}/{"wit_collapsed" if COLLAPSE_TEXT else "wit"}/')
            dataset.save_to_disk(f'{FINAL_PATH}/{"wit_collapsed" if COLLAPSE_TEXT else "wit"}/',
                                 max_shard_size=SAVE_DISK_SHARD_SIZE,
                                 num_proc=SAVE_NUM_PROC)

        if STAGE in [4, 0]:
            # Unreliable for large datasets, might have to retry a few times (automatically resumes from checkpoint):
            name = "wit_collapsed" if COLLAPSE_TEXT else "wit"
            dataset = datasets.load_from_disk(f'{FINAL_PATH}/{name}/')
            dataset.push_to_hub(f'theodor1289/{name}', max_shard_size=UPLOAD_SHARD_SIZE, private=False)

            print(f"dataset.cleanup_cache_files(): {dataset.cleanup_cache_files()}")
