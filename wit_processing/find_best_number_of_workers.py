from time import time
from typing import List, Dict, Any

from datasets import load_dataset
from torch.utils.data import DataLoader

from alkmi.definitions import VL_MAX_LENGTH_DEFAULT
from alkmi.models.flava import FlavaProcessor

processor = FlavaProcessor.from_pretrained("facebook/flava-full")


def _build_collator(inputs: List[Dict[str, Any]]):
    return processor(
        text=[i['text'] for i in inputs] if 'text' in inputs[0] else None,
        images=[i['image'] for i in inputs] if 'image' in inputs[0] else None,
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


for num_workers in [1] + list(range(2, 16, 2)):
    dataset = load_dataset("theodor1289/wit_tiny", split='train', use_auth_token=True)
    train_loader = DataLoader(dataset, shuffle=True, num_workers=num_workers,
                              batch_size=16, sampler=None, drop_last=True,
                              collate_fn=_build_collator)
    times = []
    for j in range(3):
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader):
                pass
        end = time()
        times.append(end - start)
    print(f"Finish with:{round(sum(times) / len(times), 2)}s avg., min={min(times)}, max={max(times)}, "
          f"num_workers={num_workers}")

# Finish with:10.38 avg., min=9.49, max=12.04, num_workers=1
# Finish with:5.04 avg., min=4.95, max=5.09, num_workers=2
# Finish with:3.01 avg., min=2.97, max=3.06, num_workers=4 -> good option!
# Finish with:2.67 avg., min=2.61, max=2.75, num_workers=6
# Finish with:2.49 avg., min=2.45, max=2.54, num_workers=8
# Finish with:2.60 avg., min=2.46, max=2.68, num_workers=10
# Finish with:2.86 avg., min=2.81, max=2.94, num_workers=12
# Finish with:3.13 avg., min=3.13, max=3.14, num_workers=14
