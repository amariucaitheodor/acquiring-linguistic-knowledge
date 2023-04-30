import logging

from transformers import BertForMaskedLM, FlavaForPreTraining

from pretraining.callbacks.bert_lm import BertLM
from pretraining.callbacks.flava_lm import FlavaLM

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

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


class BlimpEvalCallback(Callback):
    @torch.no_grad()
    def on_validation_start(self, trainer, pl_module) -> None:
        logger.info("Starting BLiMP Eval")
        tasks = ["anaphor_agreement.json", "argument_structure.json", "binding.json",
                 "control_raising.json", "determiner_noun_agreement.json", "ellipsis.json",
                 "filler_gap.json", "irregular_forms.json", "island_effects.json",
                 "npi_licensing.json", "quantifiers.json", "subject_verb_agreement.json"]

        if type(pl_module.model) == BertForMaskedLM:
            eval_class = BertLM
        elif type(pl_module.model) == FlavaForPreTraining:
            eval_class = FlavaLM
        else:
            raise ValueError(f"Model {type(pl_module.model)} not supported for BLiMP eval")

        accuracies = []

        for task in tasks:
            task_accuracy = accuracy_on_task(
                task_name=f"blimp_from_file:../../lm-evaluation-harness/filter-data/blimp_filtered/{task}",
                eval_model=eval_class(model=pl_module.model, batch_size=trainer.val_dataloaders.loaders[0].batch_size),
                template_name="null_prompt",
                num_fewshot=0)
            accuracies.append(round(task_accuracy * 100, 2))
            self.log(
                f"evaluation/blimp/{task.split('.json')[0]}",
                accuracies[-1],
                prog_bar=True,
                logger=True,
                rank_zero_only=True,
                sync_dist=True,
            )

        self.log(
            f"evaluation/blimp/average",
            sum(accuracies) / len(accuracies),
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
            sync_dist=True,
        )
