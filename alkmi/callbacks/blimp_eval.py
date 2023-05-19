import logging
import time
from datetime import timedelta

from transformers import BertForMaskedLM, RobertaForMaskedLM

from alkmi.callbacks.text_lm import TextLM
from alkmi.callbacks.flava_lm import FlavaLM
from models.flava import FlavaForPreTraining

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.WARNING)

import torch

from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only

import importlib

lm_eval = importlib.import_module(name="lm_eval", package="lm-evaluation-harness")


@rank_zero_only
def accuracy_on_task(task_name, eval_model, template_name, num_fewshot):
    eval_task = lm_eval.get_task_list(task_name, template_names=[template_name])
    results = lm_eval.evaluate(model=eval_model, tasks=eval_task, seed=12, num_fewshot=num_fewshot)
    accuracy = results['results'][0]['acc']
    return accuracy


class LMEvalHarnessCallback(Callback):
    def log_metric(self, name: str, value):
        self.log(name, value, prog_bar=True, logger=True, rank_zero_only=True, sync_dist=True)

    @torch.no_grad()
    def on_validation_start(self, trainer, pl_module) -> None:
        print("Starting LM Evaluation Harness")
        start = time.time()

        if type(pl_module.model) in [BertForMaskedLM, RobertaForMaskedLM]:
            eval_model = TextLM(model=pl_module.model,
                                batch_size=trainer.val_dataloaders.loaders[0].batch_size)
        elif type(pl_module.model) == FlavaForPreTraining:
            eval_model = FlavaLM(model=pl_module.model,
                                 batch_size=trainer.val_dataloaders.loaders[0].batch_size)
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
                task_name = f"blimp_from_file:./callbacks/lm-evaluation-harness/filter-data/blimp_filtered/{task}" \
                    if group_title == "blimp" else \
                    f"{task}:../lm-evaluation-harness/filter-data/glue_filtered/{task if task != 'mnli_mismatched' else 'mnli'}"
                template_name = "null_prompt" if group_title == "blimp" else lm_eval.list_templates(task)[0]
                metric_name = task.split('.json')[0] if group_title == "blimp" else task

                # Get accuracy
                task_accuracy = accuracy_on_task(
                    task_name=task_name,
                    eval_model=eval_model,
                    template_name=template_name,
                    num_fewshot=0
                )
                accuracies.append(round(task_accuracy * 100, 2))
                self.log_metric(name=f"evaluation/{group_title}/{metric_name}", value=accuracies[-1])

            self.log_metric(name=f"evaluation/{group_title}/average", value=sum(accuracies) / len(accuracies))

        print(f"Ending LM Evaluation Harness (duration: {timedelta(seconds=time.time() - start)})")