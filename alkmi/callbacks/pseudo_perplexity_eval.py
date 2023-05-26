import time
from datetime import timedelta

import torch
from datasets import load_dataset
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm
from transformers import BertForMaskedLM, RobertaForMaskedLM, PreTrainedModel

from callbacks.utils import get_corresponding_tokenizer_for_model
from data.utils import collapse_text_columns
from models.flava import FlavaForPreTraining


@rank_zero_only
def get_mlm_loss(enable_progress_bar: bool, phrases_count: int, model: PreTrainedModel, text):
    tokenizer = get_corresponding_tokenizer_for_model(model)

    mlm_loss = torch.zeros(1, dtype=torch.float64)
    for phrase in tqdm(text, desc="Pseudo-Perplexity Evaluation", total=phrases_count, unit="phrase",
                       disable=not enable_progress_bar):
        tensor_input = tokenizer(phrase,
                                 truncation=True,
                                 max_length=200,  # input size is squared! (experimentally, 200 fits in memory)
                                 return_tensors='pt')['input_ids']
        repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
        mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
        masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
        labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
        with torch.no_grad():
            if type(model) in [BertForMaskedLM, RobertaForMaskedLM]:
                mlm_loss = model(input_ids=masked_input.to("cuda:0"),
                                 labels=labels.to("cuda:0"),
                                 return_dict=True).loss
            elif type(model) == FlavaForPreTraining:
                mlm_loss = model(input_ids_masked=masked_input.to("cuda:0"),
                                 mlm_labels=labels.to("cuda:0"),
                                 return_dict=True,
                                 return_loss=True).loss_info.mlm
            else:
                raise ValueError(f"Model {type(model)} not supported for pseudo-perplexity evaluation")
        mlm_loss += mlm_loss.item()
    return mlm_loss


class PseudoPerplexityCallback(Callback):
    def __init__(self,
                 key: str,
                 split: str,
                 limit_val_batches: int,
                 enable_progress_bar: bool,
                 ):
        super().__init__()

        self.dataset = load_dataset(key, split=split, use_auth_token=True, num_proc=16)
        self.dataset = collapse_text_columns(self.dataset, purpose_msg="PPL Evaluation", need_images=False, num_proc=16)
        self.limit_val_batches = limit_val_batches
        self.enable_progress_bar = enable_progress_bar

    @torch.no_grad()
    def on_validation_start(self, trainer: Trainer, pl_module) -> None:
        """
        We use the pseudo-perplexity (PPPL) of an MLM as an intrinsic measure of how well it models a corpus
        of sentences. This differs from conventional (causal) LMs, which use perplexity (PPL).

        References: Section 2.3 of https://arxiv.org/pdf/1910.14659.pdf
        """
        print("Starting Pseudo-Perplexity Evaluation")
        start = time.time()

        phrases_count = self.limit_val_batches
        text = self.dataset[:phrases_count]["text"]

        mlm_loss = get_mlm_loss(self.enable_progress_bar, phrases_count, pl_module.model, text)

        ppl = torch.exp(mlm_loss / phrases_count).item()
        self.log("evaluation/pseudo_perplexity", ppl, prog_bar=True, logger=True, rank_zero_only=True, sync_dist=True)

        print(f"Ending Pseudo-Perplexity Evaluation (PPL: {ppl}) (duration: {timedelta(seconds=time.time() - start)})")
