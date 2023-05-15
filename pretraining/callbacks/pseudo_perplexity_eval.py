import time
from datetime import timedelta

import torch
from datasets import load_dataset
from pytorch_lightning import Callback
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForMaskedLM, FlavaForPreTraining, RobertaForMaskedLM

from data.utils import collapse_wit_text, WIT_ALT_TEXT_COLUMNS


class PseudoPerplexityCallback(Callback):
    def __init__(self, key: str, limit_val_batches: int, text_collapse_batch_size: int = 100):
        super().__init__()

        self.dataset = load_dataset(key, split="test", use_auth_token=True, num_proc=32) \
            .remove_columns(["image"])
        self.dataset = self.dataset.map(
            collapse_wit_text,
            batched=True,
            num_proc=16,
            batch_size=text_collapse_batch_size,
            remove_columns=WIT_ALT_TEXT_COLUMNS
        )
        self.limit_val_batches = limit_val_batches
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    @torch.no_grad()
    def on_validation_start(self, trainer, pl_module) -> None:
        """
        We use the pseudo-perplexity (PPPL) of FLAVA (an MLM) as an intrinsic measure of how well it models a corpus
        of sentences. This differs from conventional (causal) LMs, which use perplexity (PPL).

        References: Section 2.3 of https://arxiv.org/pdf/1910.14659.pdf
        """
        print("Starting Pseudo-Perplexity Evaluation")
        start = time.time()

        phrases_count = self.limit_val_batches
        self.text = self.dataset[:phrases_count]["text"]

        avg_mlm_loss = torch.zeros(1, dtype=torch.float64)
        for phrase in tqdm(self.text, desc="Pseudo-Perplexity Evaluation", total=phrases_count, unit="phrase"):
            tensor_input = self.tokenizer(phrase,
                                          truncation=True,
                                          max_length=200,  # input size is squared! (experimentally, 200 fits in memory)
                                          return_tensors='pt')['input_ids']
            repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
            mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
            masked_input = repeat_input.masked_fill(mask == 1, self.tokenizer.mask_token_id)
            labels = repeat_input.masked_fill(masked_input != self.tokenizer.mask_token_id, -1)
            with torch.inference_mode():
                if type(pl_module.model) in [BertForMaskedLM, RobertaForMaskedLM]:
                    mlm_loss = pl_module.model(input_ids=masked_input.to("cuda:0"),
                                               labels=labels.to("cuda:0"),
                                               return_dict=True).loss
                elif type(pl_module.model) == FlavaForPreTraining:
                    mlm_loss = pl_module.model(input_ids_masked=masked_input.to("cuda:0"),
                                               mlm_labels=labels.to("cuda:0"),
                                               return_dict=True,
                                               return_loss=True).losses.mlm_loss
                else:
                    raise ValueError(f"Model {type(pl_module.model)} not supported for pseudo-perplexity evaluation")
            avg_mlm_loss += mlm_loss.item()

        ppl = torch.exp(avg_mlm_loss / phrases_count).item()
        self.log("evaluation/pseudo_perplexity", ppl, prog_bar=True, logger=True, rank_zero_only=True, sync_dist=True)

        print(f"Ending Pseudo-Perplexity Evaluation (PPL: {ppl}) (duration: {timedelta(seconds=time.time() - start)})")
