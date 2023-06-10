import time
from datetime import timedelta

import torch
from datasets import load_dataset
from pytorch_lightning import Callback, Trainer
from tqdm import tqdm
from transformers import BertForMaskedLM, RobertaForMaskedLM

from callbacks.utils import get_corresponding_tokenizer_for_model
from data.utils import collapse_text_columns
from models.flava import FlavaForPreTraining


class PseudoPerplexityCallback(Callback):
    def __init__(self,
                 key: str,
                 split: str,
                 limit_val_batches: int,
                 enable_progress_bar: bool,
                 ):
        super().__init__()

        print(f"[PPL Evaluation] Loading dataset '{key}' with split '{split}'")
        self.dataset = load_dataset(key, split=split, use_auth_token=True, num_proc=16)
        self.dataset = collapse_text_columns(self.dataset, purpose_msg="PPL Evaluation", need_images=False)
        print(f"[PPL Evaluation] Length of the dataset is {len(self.dataset)}")
        self.limit_val_batches = limit_val_batches
        self.enable_progress_bar = enable_progress_bar

    @torch.no_grad()
    def on_validation_start(self, trainer: Trainer, pl_module) -> None:
        """
        We use the pseudo-perplexity (PPPL) of an MLM as an intrinsic measure of how well it models a corpus
        of sentences. This differs from conventional (causal) LMs, which use perplexity (PPL).

        References: Section 2.3 of https://arxiv.org/pdf/1910.14659.pdf
        """
        # I tried a different logic here, but could only get rank 0 to log metrics, so I switched back
        idx_start = trainer.global_rank * self.limit_val_batches
        idx_end = (trainer.global_rank + 1) * self.limit_val_batches

        print(f"[PPL Evaluation] Starting from index {idx_start} to index {idx_end}.")
        model_device = next(pl_module.model.parameters()).device
        start = time.time()

        vram_gb = round(torch.cuda.mem_get_info()[1] / (1024 ** 3), 2)
        max_phrase_length = 200 if vram_gb > 20 else (300 if vram_gb > 30 else 100)
        print(f"[PPL Evaluation] Based on the available VRAM ({vram_gb}GBs), "
              f"we will use a max phrase length of {max_phrase_length}.")

        text = self.dataset[idx_start:idx_end]["text"]
        tokenizer = get_corresponding_tokenizer_for_model(pl_module.model)
        total_mlm_loss = torch.zeros(1, dtype=torch.float64)
        for phrase in tqdm(text,
                           desc="Pseudo-Perplexity Evaluation",
                           total=self.limit_val_batches, unit="phrase",
                           disable=not self.enable_progress_bar):
            tensor_input = tokenizer(phrase, truncation=True,
                                     max_length=max_phrase_length, return_tensors='pt')['input_ids']
            repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
            mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]

            masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
            labels = repeat_input.masked_fill(masked_input != tokenizer.mask_token_id, -100)
            with torch.no_grad():
                if type(pl_module.model) in [BertForMaskedLM, RobertaForMaskedLM]:
                    mlm_loss = pl_module.model(input_ids=masked_input.to(model_device),
                                               labels=labels.to(model_device),
                                               return_dict=True).loss
                elif type(pl_module.model) == FlavaForPreTraining:
                    mlm_loss = pl_module.model(input_ids_masked=masked_input.to(model_device),
                                               mlm_labels=labels.to(model_device),
                                               return_dict=True,
                                               return_loss=True).loss_info.mlm
                else:
                    raise ValueError(f"[PPL Evaluation] Model {type(pl_module.model)} not supported.")

            if torch.isnan(mlm_loss):
                print(f"[PPL Evaluation] WARNING: MLM loss is NaN for phrase '{phrase}', masked_input {masked_input}. "
                      f"Skipping it.")
            else:
                total_mlm_loss += mlm_loss.item()

        print(f"[PPL Evaluation] Computing e^({total_mlm_loss.item()} / {self.limit_val_batches})")
        ppl = torch.exp(total_mlm_loss / self.limit_val_batches).item()

        self.log("evaluation/pseudo_perplexity", ppl, prog_bar=True, logger=True, rank_zero_only=False, sync_dist=True)
        print(f"[PPL Evaluation] Ending with PPL={ppl} (duration: {timedelta(seconds=time.time() - start)})")
