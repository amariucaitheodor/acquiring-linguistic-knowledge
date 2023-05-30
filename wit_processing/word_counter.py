import evaluate
from datasets import load_dataset

from alkmi.data.utils import WIT_OTHER_TEXT_COLUMNS
from wit_processing.initial_setup import process_text

dataset = load_dataset("facebook/pmd", "wit", split='train', use_auth_token=True, num_proc=48)
dataset = dataset.map(process_text, batched=True, num_proc=16, batch_size=100,
                      fn_kwargs={"num_threads": 64, "collapse": False},
                      remove_columns=["meta", "source"])
print(dataset.column_names)
# ['image_url', 'image', 'text', 'context_page_description',
# 'context_section_description', 'caption_alt_text_description', 'caption_attribution_description']

total_words = 0
strings = []
for column in ['text'] + WIT_OTHER_TEXT_COLUMNS:
    wordcount = evaluate.load("word_count")
    text = list(filter(lambda item: item is not None, dataset[column]))
    results = wordcount.compute(data=text)
    print(column)
    print(f"Total words: {results['total_word_count']}, "
          f"No. of duplicates: {results['total_word_count'] - results['unique_words']}, "
          f"No. of unique: {results['unique_words']}")
    total_words += results['total_word_count']
print(f"Total words: {total_words}")

# For Facebook/PMD/WiT (total 1.299.915.533 words):
# - **caption** Total words: 50.554.233, No. of duplicates: 49.433.647, No. of unique: 1.120.586
# - **context_page_description** Total words: 424.693.339, No. of duplicates: 423.317.941, No. of unique: 1.375.398
# - **context_section_description** Total words: 723.116.439, No. of duplicates: 720.014.213, No. of unique: 3.102.226
# - **caption_alt_text_description** Total words: 3.757.265, No. of duplicates: 3.513.898, No. of unique: 243.367
# - **caption_attribution_description** Total words: 97.794.257, No. of duplicates: 95.512.148, No. of unique: 2.282.109
