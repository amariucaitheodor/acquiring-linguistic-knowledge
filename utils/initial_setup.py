import io
import json
import time
import urllib
from concurrent.futures import ThreadPoolExecutor

import PIL.Image
import datasets
from PIL import PngImagePlugin
from datasets import Image
from datasets.utils.file_utils import get_datasets_user_agent
from transformers.image_transforms import resize
from transformers.image_utils import PILImageResampling
import numpy as np

from alkmi.data.utils import WIT_OTHER_TEXT_COLUMNS

USER_AGENT = get_datasets_user_agent()

FLAVA_IMAGE_SIZE = (224, 224)
NUM_THREADS = 1
VERSION_ROWS = {"tiny": 1_000, "medium": 38_200, "full": -1}
VERSION = "full"


def fetch_single_image(image_url, timeout=None, retries: int = 2):
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(image_url, data=None, headers={"user-agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
                if image.mode != 'RGB':
                    image = image.convert("RGBA")
                    white_fill_color = (255, 255, 255)
                    background = PIL.Image.new(image.mode[:-1], image.size, white_fill_color)
                    background.paste(image, image.split()[-1])  # omit transparency
                    image = background
                    image.convert("RGB")
                image = np.array(image)
                image = resize(image, size=FLAVA_IMAGE_SIZE, resample=PILImageResampling.BICUBIC, return_numpy=False)
            break
        except Exception as e:
            exception_string = e.__str__()
            if not exception_string.startswith("HTTP Error 404"):
                if exception_string.__contains__("HTTP Error 429"):
                    print("Sleeping for 10 seconds... (Too Many Requests)")
                    time.sleep(10)
                elif exception_string.__contains__("[Errno 110] Connection timed out") \
                        or exception_string.__contains__("[Errno 104] Connection reset by peer"):
                    print("Sleeping for 5 seconds... (Connection Trouble)")
                    time.sleep(5)
                else:
                    print(f"Error fetching image (other than 404): {e} ({image_url})")
            image = None
    return image


def fetch_images(batch, num_threads: int):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(fetch_single_image, batch["image_url"]))
    return batch


def process_text(batch, num_threads: int):
    # This extracts the alternative text fields from the "meta" field
    with ThreadPoolExecutor(max_workers=num_threads) as executor:  # "text" is the caption
        for field in WIT_OTHER_TEXT_COLUMNS:
            batch[field] = list(executor.map(lambda meta: json.loads(meta)[field], batch["meta"]))
    return batch


text_and_image_both_present = lambda examples: [text is not None and image is not None for text, image in
                                                zip(examples["text"], examples["image"])]

if __name__ == "__main__":
    if VERSION != "full":
        dataset = datasets.load_dataset("facebook/pmd", "wit", split='train', use_auth_token=True, num_proc=48) \
            .select(list(range(VERSION_ROWS[VERSION])))
        dataset = dataset.map(fetch_images,
                              fn_kwargs={"num_threads": NUM_THREADS},
                              batched=True,
                              batch_size=20,
                              num_proc=20)
        dataset = dataset.filter(
            text_and_image_both_present,
            batched=True,
            num_proc=6,
            batch_size=25,
            desc=f"Filtering out data with missing content from wit_{VERSION}"
        )
        dataset = dataset.train_test_split(test_size=0.1, train_size=0.9)  # First split, then collapse text
        for split in ['train', 'test']:
            print("Processing split", split)
            dataset[split] = dataset[split].map(process_text,
                                                batched=True,
                                                num_proc=24,
                                                batch_size=100,
                                                fn_kwargs={"num_threads": NUM_THREADS * 2},
                                                remove_columns=["meta", "source"])
        dataset.push_to_hub(f"theodor1289/wit_{VERSION}", max_shard_size="500MB", private=False)
    else:
        STAGE = 4
        SAVE_DISK_SHARD_SIZE, UPLOAD_SHARD_SIZE, SAVE_NUM_PROC = "10GB", "500MB", 32
        SCRATCH_PATH, FINAL_PATH = '/cluster/scratch/tamariucai', '/cluster/work/cotterell/tamariucai'
        PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 ** 2)
        Image.MAX_IMAGE_PIXELS = None

        if STAGE in [1, 0]:
            print(f"Running STEP 1: get WiT, process the images and save to disk")
            dataset = datasets.load_dataset("facebook/pmd", "wit", split='train', use_auth_token=True, num_proc=12)
            dataset = dataset.map(fetch_images,
                                  fn_kwargs={"num_threads": NUM_THREADS},
                                  batched=True,
                                  batch_size=20,
                                  num_proc=20,
                                  desc=f"Downloading the images from facebook/pmd")
            dataset.save_to_disk(f'{FINAL_PATH}/wit_images/', max_shard_size=SAVE_DISK_SHARD_SIZE,
                                 num_proc=SAVE_NUM_PROC)
            print(f"dataset.cleanup_cache_files(): {dataset.cleanup_cache_files()}")

        if STAGE in [2, 0]:
            print(f"Running STEP 2: load from disk, filter missing content, save to disk")
            dataset = datasets.load_from_disk(f'{FINAL_PATH}/wit_images/')
            dataset = dataset.cast_column("image", Image(decode=False))  # MUCH faster processing
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
            dataset = dataset.cast_column("image", Image(decode=False))  # MUCH faster processing
            dataset = dataset.train_test_split(test_size=0.1, train_size=0.9)

            print(f"Midpoint STEP 3: processing the text AFTER splitting the dataset...")
            for split in ['train', 'test']:
                print("Processing split", split)
                dataset[split] = dataset[split].map(process_text,
                                                    batched=True,
                                                    num_proc=24,
                                                    batch_size=100,
                                                    fn_kwargs={"num_threads": NUM_THREADS * 2},
                                                    remove_columns=["meta", "source"],
                                                    desc=f"Processing the text for split '{split}'")

            print(f'Final STEP 3: saving to disk at {FINAL_PATH}/wit/')
            dataset.save_to_disk(f'{FINAL_PATH}/wit/', max_shard_size=SAVE_DISK_SHARD_SIZE, num_proc=SAVE_NUM_PROC)

        if STAGE in [4, 0]:
            # Unreliable for large datasets, might have to retry a few times (automatically resumes from checkpoint):
            dataset = datasets.load_from_disk(f'{FINAL_PATH}/wit/')
            dataset.push_to_hub(f'theodor1289/wit', max_shard_size=UPLOAD_SHARD_SIZE, private=False)

            print(f"dataset.cleanup_cache_files(): {dataset.cleanup_cache_files()}")
