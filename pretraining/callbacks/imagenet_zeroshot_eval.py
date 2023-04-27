# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from transformers import FlavaForPreTraining, FlavaProcessor

from callbacks.imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template

logging.basicConfig(level=logging.INFO)

import torch

from pytorch_lightning import Callback, LightningDataModule
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _zero_shot_classifier(model: FlavaForPreTraining, device, processor: FlavaProcessor):
    zeroshot_weights = []
    max_length = max(
        [len(template(classname)) for template in openai_imagenet_template for classname in imagenet_classnames])

    for classname in tqdm(imagenet_classnames[:16]):
        texts = [template(classname) for template in openai_imagenet_template]
        input_ids = processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
        )["input_ids"]
        class_embeddings = model(input_ids=input_ids.to(device)).text_embeddings
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        zeroshot_weights.append(class_embedding)

    zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def _accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


@rank_zero_only
def run_imagenet_zero_shot(model: FlavaForPreTraining, dataloader, device, processor: FlavaProcessor):
    logger.info("Starting ImageNet Zero-Shot Eval")
    logger.info("Building classifier")
    classifier = _zero_shot_classifier(model, device, processor)
    logger.info("Classifier built")
    top1, top5, n = 0.0, 0.0, 0.0
    for sample in tqdm(dataloader):
        sample = sample[0]
        images = sample["image"]
        target = sample["label"]

        outputs = processor(
            images=images,
            return_tensors="pt",
            return_image_mask=True,
            return_codebook_pixels=True,
            return_dict=True,
        ).to(device)
        image_features = model(**outputs).image_embeddings
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ classifier

        # measure accuracy
        acc1, acc5 = _accuracy(logits, target, topk=(1, 5))
        top1 += acc1
        top5 += acc5
        n += images.size(0)

    top1 = top1 / n
    top5 = top5 / n
    return {"zeroshot-val-top1": top1, "zeroshot-val-top5": top5}


class MultimodalEvalCallback(Callback):
    def __init__(self, imagenet_datamodule: LightningDataModule):
        super().__init__()
        self.imagenet_val_dataloader = imagenet_datamodule.val_dataloader()
        self.processor = FlavaProcessor.from_pretrained("facebook/flava-full")

    @torch.no_grad()
    def on_validation_start(self, trainer, pl_module, **kwargs) -> None:
        metrics = run_imagenet_zero_shot(
            pl_module.model,
            self.imagenet_val_dataloader,
            pl_module.device,
            self.processor
        )
        if metrics is not None:
            for key in metrics:
                self.log(
                    f"evaluation/imagenet/{key}",
                    metrics[key],
                    prog_bar=True,
                    logger=True,
                    rank_zero_only=True,
                    sync_dist=True,
                )
