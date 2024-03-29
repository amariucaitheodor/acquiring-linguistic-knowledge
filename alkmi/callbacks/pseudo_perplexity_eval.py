import os
import time
from datetime import timedelta, datetime

import torch
from datasets import load_dataset, DownloadMode
from pytorch_lightning import Callback, Trainer
from tqdm import tqdm
from transformers import BertForMaskedLM, RobertaForMaskedLM

from callbacks.utils import get_corresponding_tokenizer_for_model, replace_flava_submodel_with_orig_for_eval
from data.utils import collapse_text_columns
from alkmi.models.flava import FlavaForPreTraining


class PseudoPerplexityCallback(Callback):
    def __init__(self,
                 key: str,
                 split: str,
                 limit_val_batches: int,
                 enable_progress_bar: bool,
                 ):
        super().__init__()

        self.tokenizer = None
        print(f"[PPL Evaluation] Loading dataset '{key}' with split '{split}'")
        self.ppl_dataset = load_dataset(key,
                                        split=split,
                                        use_auth_token=True,
                                        num_proc=32,
                                        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                                        save_infos=True)
        self.ppl_dataset.__setattr__("collapse_id", f"{key}-{split}".replace(":", "").replace("/", ""))
        self.ppl_dataset = collapse_text_columns(self.ppl_dataset, need_images=False)
        print(f"[PPL Evaluation] Length of the dataset is {len(self.ppl_dataset)}")
        self.total_phrases = limit_val_batches
        self.enable_progress_bar = enable_progress_bar

    @torch.no_grad()
    def on_validation_start(self, trainer: Trainer, pl_module) -> None:
        """
        We use the pseudo-perplexity (PPPL) of an MLM as an intrinsic measure of how well it models a corpus
        of sentences. This differs from conventional (causal) LMs, which use perplexity (PPL).

        References: Section 2.3 of https://arxiv.org/pdf/1910.14659.pdf
        """
        # I tried a different logic here, but could only get rank 0 to log metrics, so I switched back
        idx_start = trainer.global_rank * self.total_phrases
        idx_end = (trainer.global_rank + 1) * self.total_phrases

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
        batch_size = trainer.val_dataloaders.loaders[0].batch_size // 2  # should prevent OOM

        print(f"[PPL Evaluation] Starting from index {idx_start} to index {idx_end}.")
        if type(pl_module.model) == FlavaForPreTraining:
            optimized_text_model = replace_flava_submodel_with_orig_for_eval(pl_module.model)
        pl_module.model.eval()
        start = time.time()

        if not self.tokenizer:
            self.tokenizer = get_corresponding_tokenizer_for_model(pl_module.model)
        text = self.ppl_dataset[idx_start:idx_end]["text"]
        total_mlm_loss = torch.zeros(1, dtype=torch.float64)
        for phrase in tqdm(text,
                           desc="Pseudo-Perplexity Evaluation",
                           total=self.total_phrases,
                           unit="phrase",
                           disable=not self.enable_progress_bar):
            tensor_input = self.tokenizer(phrase,
                                          truncation=True,
                                          max_length=300,  # to prevent OOM
                                          return_tensors='pt')['input_ids']
            repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
            mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]

            masked_input = repeat_input.masked_fill(mask == 1, self.tokenizer.mask_token_id)
            labels = repeat_input.masked_fill(masked_input != self.tokenizer.mask_token_id, -100)

            phrase_unnormalized_loss = torch.zeros(1, dtype=torch.float64)
            phrase_length = masked_input.shape[0]
            batched_phrase = torch.split(torch.stack((masked_input, labels)), batch_size)

            with torch.no_grad():
                for batch in batched_phrase:
                    masked_input, labels = batch[0], batch[1]

                    if type(pl_module.model) in [BertForMaskedLM, RobertaForMaskedLM]:
                        mlm_loss = pl_module.model(input_ids=masked_input.to(pl_module.device),
                                                   labels=labels.to(pl_module.device),
                                                   return_dict=True).loss
                    elif type(pl_module.model) == FlavaForPreTraining:
                        mlm_loss = pl_module.model(input_ids_masked=masked_input.to(pl_module.device),
                                                   mlm_labels=labels.to(pl_module.device),
                                                   return_dict=True,
                                                   return_loss=True).loss_info.mlm
                    else:
                        raise ValueError(f"[PPL Evaluation] Model {type(pl_module.model)} not supported.")

                    if torch.isnan(mlm_loss):
                        print(f"[PPL Evaluation] WARNING: MLM loss is NaN for phrase '{phrase}', "
                              f"masked_input {masked_input}. Skipping it.")
                    else:
                        phrase_unnormalized_loss += mlm_loss.item() * masked_input.shape[0]

            total_mlm_loss += phrase_unnormalized_loss / phrase_length

        print(f"[PPL Evaluation {datetime.now()}] "
              f"Computing e^({total_mlm_loss} / {self.total_phrases})")
        ppl = torch.exp(total_mlm_loss / self.total_phrases).item()
        self.log("evaluation/pseudo_perplexity", ppl, prog_bar=True, logger=True, rank_zero_only=False, sync_dist=True)

        if type(pl_module.model) == FlavaForPreTraining:
            pl_module.model.flava.text_model = optimized_text_model
        pl_module.model.train()
        print(f"[PPL Evaluation {datetime.now()}] "
              f"Ending with PPL={ppl} (duration: {timedelta(seconds=time.time() - start)})")
