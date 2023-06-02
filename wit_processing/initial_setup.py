import io
import json
import urllib
from concurrent.futures import ThreadPoolExecutor

import PIL.Image
import datasets
from PIL import PngImagePlugin
from datasets import Image
from datasets.utils.file_utils import get_datasets_user_agent
from transformers.image_transforms import resize
from transformers.image_utils import PILImageResampling

from alkmi.data.utils import WIT_OTHER_TEXT_COLUMNS

USER_AGENT = get_datasets_user_agent()


def fetch_single_image(image_url, timeout=None, retries: int = 3):
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
                flava_image_size = (224, 224)
                image = resize(image, size=flava_image_size, resample=PILImageResampling.BICUBIC)
            break
        except Exception:
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
    NUM_THREADS = 64
    MINI_VERSION = False

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
            desc=f"Filtering out data with missing content from wit_tiny"
        )
        dataset = dataset.train_test_split(test_size=0.1, train_size=0.9)  # First split, then collapse text
        for split in ['train', 'test']:
            print("Processing split", split)
            dataset[split] = dataset[split].map(process_text, batched=True, num_proc=24, batch_size=100,
                                                fn_kwargs={"num_threads": NUM_THREADS},
                                                remove_columns=["meta", "source"])
        dataset.push_to_hub(f"theodor1289/wit", max_shard_size="500MB", private=False)
    else:
        STAGE = 1
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
                                  num_proc=16,
                                  remove_columns=["image_url"])
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
                dataset[split] = dataset[split].map(process_text, batched=True, num_proc=24, batch_size=100,
                                                    fn_kwargs={"num_threads": NUM_THREADS},
                                                    remove_columns=["meta", "source"])

            print(f'Final STEP 3: saving to disk at {FINAL_PATH}/wit/')
            dataset.save_to_disk(f'{FINAL_PATH}/wit/', max_shard_size=SAVE_DISK_SHARD_SIZE, num_proc=SAVE_NUM_PROC)

        if STAGE in [4, 0]:
            # Unreliable for large datasets, might have to retry a few times (automatically resumes from checkpoint):
            dataset = datasets.load_from_disk(f'{FINAL_PATH}/wit/')
            dataset.push_to_hub(f'theodor1289/wit', max_shard_size=UPLOAD_SHARD_SIZE, private=False)

            print(f"dataset.cleanup_cache_files(): {dataset.cleanup_cache_files()}")
