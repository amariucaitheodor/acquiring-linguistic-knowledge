import os
import time
from datetime import timedelta

from transformers import BertForMaskedLM, RobertaForMaskedLM

from alkmi.callbacks.text_lm import TextLM
from alkmi.callbacks.flava_lm import FlavaLM
from callbacks.utils import replace_flava_submodel_with_orig_for_eval
from alkmi.models.flava import FlavaForPreTraining

import torch

from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only

import importlib

lm_eval = importlib.import_module(name="lm_eval", package="evaluation-pipeline")


class LMEvalHarnessCallback(Callback):

    def __init__(self, enable_progress_bar: bool):
        super().__init__()
        self.enable_progress_bar = enable_progress_bar
        self.eval_wrapper = None

    def log_metric(self, name: str, value):
        self.log(name, value, prog_bar=True, logger=True, rank_zero_only=True, sync_dist=True)

    @torch.no_grad()
    @rank_zero_only
    def on_validation_start(self, trainer, pl_module) -> None:
        print("Starting LM Evaluation Harness")
        start = time.time()
        if type(pl_module.model) == FlavaForPreTraining:
            optimized_text_model = replace_flava_submodel_with_orig_for_eval(pl_module.model)
        pl_module.model.eval()

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

        if self.eval_wrapper is None:
            if type(pl_module.model) in [BertForMaskedLM, RobertaForMaskedLM]:
                self.eval_wrapper = TextLM(model=pl_module.model,
                                           batch_size=trainer.val_dataloaders.loaders[0].batch_size,
                                           enable_progress_bar=self.enable_progress_bar)
            elif type(pl_module.model) == FlavaForPreTraining:
                self.eval_wrapper = FlavaLM(model=pl_module.model,
                                            batch_size=trainer.val_dataloaders.loaders[0].batch_size,
                                            enable_progress_bar=self.enable_progress_bar)
            else:
                raise ValueError(f"Model {type(pl_module.model)} not supported for BLiMP eval")

        TASKS = {
            "glue": ["cola", "sst", "mrpc", "qqp", "mnli", "mnli_mismatched", "qnli", "rte",
                     "boolq", "multirc", "wsc"],
            "blimp": ["anaphor_agreement.json", "argument_structure.json", "binding.json",
                      "control_raising.json", "determiner_noun_agreement.json", "ellipsis.json",
                      "filler_gap.json", "irregular_forms.json", "island_effects.json",
                      "npi_licensing.json", "quantifiers.json", "subject_verb_agreement.json"],
        }

        for group_title in ['blimp']:  # 'glue' requires fine-tuning before eval, hence it gives ~50% accuracy on MLM!
            print(f"Running on {group_title}...")

            accuracies = []
            for task in TASKS[group_title]:
                # Setup
                task_name = f"blimp_from_file:./callbacks/evaluation-pipeline/filter-data/blimp_filtered/{task}" \
                    if group_title == "blimp" else \
                    f"{task}:../evaluation-pipeline/filter-data/glue_filtered/{task if task != 'mnli_mismatched' else 'mnli'}"
                template_name = "null_prompt" if group_title == "blimp" else lm_eval.list_templates(task)[0]
                metric_name = task.split('.json')[0] if group_title == "blimp" else task

                # Get accuracy
                eval_task = lm_eval.get_task_list(task_name, template_names=[template_name])
                results = lm_eval.evaluate(model=self.eval_wrapper, tasks=eval_task, seed=12, num_fewshot=0)
                task_accuracy = results['results'][0]['acc']

                accuracies.append(round(task_accuracy * 100, 2))
                self.log_metric(name=f"evaluation/{group_title}/{metric_name}", value=accuracies[-1])
                print(f"evaluation/{group_title}/{metric_name}: {accuracies[-1]}")

            self.log_metric(name=f"evaluation/{group_title}_average", value=sum(accuracies) / len(accuracies))
            print(f"evaluation/{group_title}_average: {sum(accuracies) / len(accuracies)}")

        if type(pl_module.model) == FlavaForPreTraining:
            pl_module.model.flava.text_model = optimized_text_model
        pl_module.model.train()
        print(f"Ending LM Evaluation Harness (duration: {timedelta(seconds=time.time() - start)})")
