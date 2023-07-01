# Copyright (c) Theodor Amariucai & Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

import torch
from datasets import load_dataset, DownloadMode
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from callbacks.imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template
from callbacks.utils import replace_flava_submodel_with_orig_for_eval
from definitions import VL_MAX_LENGTH_DEFAULT
from models.flava import FlavaForPreTraining, FlavaProcessor


class ImageNetZeroshotCallback(Callback):
    def __init__(self, enable_progress_bar: bool):
        super().__init__()

        self.enable_progress_bar = enable_progress_bar
        self.imagenet_val_dataloader = None
        self.processor = FlavaProcessor.from_pretrained("facebook/flava-full")
        self.val_dataset = load_dataset("imagenet-1k",
                                        split="validation",
                                        use_auth_token=True,
                                        num_proc=32,
                                        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                                        save_infos=True)

    def _vl_processor(self, inputs: List[Dict[str, Any]]):
        outputs = self.processor(
            text=[i['text'] for i in inputs] if 'text' in inputs[0] else None,
            images=[i['image'].convert("RGB") for i in inputs] if 'image' in inputs[0] else None,
            return_tensors="pt",
            padding="max_length",
            add_special_tokens=False,
            truncation=True,  # very important!
            max_length=VL_MAX_LENGTH_DEFAULT,
            return_image_mask=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_special_tokens_mask=False,  # not needed
            return_codebook_pixels=False,  # not needed
        )
        if 'label' in inputs[0]:
            outputs['label'] = torch.tensor([i['label'] for i in inputs])
        return outputs

    def _build_zero_shot_classifier(self, model: FlavaForPreTraining, device) -> Tensor:
        zeroshot_weights = []
        for classname in tqdm(imagenet_classnames, desc="Building classifier", unit="classes",
                              disable=not self.enable_progress_bar):
            texts = self._vl_processor(inputs=[{"text": template(classname)} for template in openai_imagenet_template])
            texts = texts.to(device)

            class_embeddings = model.flava.get_text_features(**texts)
            class_embeddings = model.flava.text_projection(class_embeddings[:, 0, :])
            class_embeddings = nn.functional.normalize(class_embeddings, dim=-1)

            mean_embedding = class_embeddings.mean(dim=0)
            mean_embedding /= mean_embedding.norm()

            zeroshot_weights.append(mean_embedding)
        return torch.stack(zeroshot_weights, dim=1).to(device)

    @torch.no_grad()
    @rank_zero_only
    def on_validation_start(self, trainer, pl_module, **kwargs) -> None:
        print(f"[ImageNet Zero-Shot Evaluation] Starting...")
        optimized_text_model = replace_flava_submodel_with_orig_for_eval(pl_module.model)
        pl_module.model.eval()
        start = time.time()

        # Below prevents:
        # F.conv2d(input, weight, bias, self.stride, [...])
        # RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
        torch.backends.cudnn.benchmark = False

        if not self.imagenet_val_dataloader:
            self.imagenet_val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=trainer.val_dataloaders.loaders[0].batch_size,
                num_workers=4,
                shuffle=False,
                # uneven batches can cause distributed issues,
                # drop last batch to prevent those.
                # ideally, we don't need to drop these for unimodal cases
                # but just to be safe
                drop_last=True,
                collate_fn=self._vl_processor,
            )

        print(f"[ImageNet Zero-Shot Evaluation] Building classifier...")
        classifier: Tensor = self._build_zero_shot_classifier(pl_module.model, pl_module.device)
        print(f"[ImageNet Zero-Shot Evaluation] Classifier built, shape is {classifier.size()}.")

        top1, top5, total_samples = 0.0, 0.0, 0.0
        for batch in tqdm(self.imagenet_val_dataloader, disable=not self.enable_progress_bar):
            batch = batch.to(pl_module.device)

            labels = batch.pop("label")
            total_samples += labels.size(0)

            image_features = pl_module.model.flava.get_image_features(**batch)
            image_features = pl_module.model.flava.image_projection(image_features[:, 0, :])
            image_features = nn.functional.normalize(image_features, dim=-1)

            logits_per_image = 100.0 * image_features @ classifier
            pred: Tensor = logits_per_image.topk(k=5, dim=1, largest=True, sorted=True)[1].t()
            correct: Tensor = pred.eq(labels.view(1, -1).expand_as(pred)).float()

            top1 += correct[:1].reshape(-1).sum(0, keepdim=True).item()  # accuracy for top1
            top5 += correct[:5].reshape(-1).sum(0, keepdim=True).item()  # accuracy for top5

        for key, value in [("top1", top1 / total_samples), ("top5", top5 / total_samples)]:
            self.log(f"evaluation/imagenet_zeroshot/{key}", value,
                     prog_bar=True, logger=True, rank_zero_only=True, sync_dist=True)

        torch.backends.cudnn.benchmark = True
        pl_module.model.flava.text_model = optimized_text_model
        pl_module.model.train()
        print(f"[ImageNet Zero-Shot Evaluation {datetime.now()}] "
              f"Ending with top5={top5 / total_samples} (duration: {timedelta(seconds=time.time() - start)})")
