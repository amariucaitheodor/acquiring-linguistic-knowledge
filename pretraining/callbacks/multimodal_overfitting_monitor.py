# Copyright Theodor Amariucai & The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
Cancel Task
^^^^^^^^^^^^^^

Monitor a multimodal validation loss and set the task weight to 0 when it stops improving.

"""
import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from lightning_fabric.utilities.rank_zero import _get_rank
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_prefixed_message, rank_zero_warn
from torch import Tensor

from data.multidata import MultiDataModule

log = logging.getLogger(__name__)


class MultimodalOverfittingMonitor(Callback):
    r"""
    Monitor a multimodal validation loss and set the task weight to 0 when it stops improving.

    Args:
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute
            change of less than or equal to `min_delta`, will count as no improvement.
        patience: number of checks with no improvement
            after which task weight will be set to 0. Under the default configuration, one check happens after
            every training epoch. However, the frequency of validation can be modified by setting various parameters on
            the ``Trainer``, for example ``check_val_every_n_epoch`` and ``val_check_interval``.

            .. note::

                It must be noted that the patience parameter counts the number of validation checks with
                no improvement, and not the number of training epochs. Therefore, with parameters
                ``check_val_every_n_epoch=10`` and ``patience=3``, the trainer will perform at least 40 training
                epochs before being stopped.

        verbose: verbosity mode.
        mode: one of ``'min'``, ``'max'``. In ``'min'`` mode, training will stop when the quantity
            monitored has stopped decreasing and in ``'max'`` mode it will stop when the quantity
            monitored has stopped increasing.
        strict: whether to crash the training if `monitor` is not found in the validation metrics.
        check_finite: When set ``True``, stops training when the monitor becomes NaN or infinite.
        log_rank_zero_only: When set ``True``, logs the status of the early stopping callback only for rank 0 process.

    Raises:
        MisconfigurationException:
            If ``mode`` is none of ``"min"`` or ``"max"``.
        RuntimeError:
            If the metric ``monitor`` is not available.

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pretraining.callbacks.multimodal_overfitting_monitor import MultimodalOverfittingMonitor
        >>> multimodal_overfitting_monitor = MultimodalOverfittingMonitor('validation/losses/itm_loss')
        >>> trainer = Trainer(callbacks=[multimodal_overfitting_monitor])

    .. tip:: Saving and restoring multiple early stopping callbacks at the same time is supported under variation in the
        following arguments:

        *monitor, mode*

        Read more: :ref:`Persisting Callback State <extensions/callbacks_state:save callback state>`
    """
    mode_dict = {"min": torch.lt, "max": torch.gt}

    order_dict = {"min": "<", "max": ">"}

    def __init__(
            self,
            monitor: str,
            datamodule: MultiDataModule,
            min_delta: float = 0.0,
            patience: int = 3,
            verbose: bool = False,
            mode: str = "min",
            strict: bool = True,
            check_finite: bool = True,
            log_rank_zero_only: bool = False,
    ):
        super().__init__()
        self.monitor = monitor
        self.datamodule: MultiDataModule = datamodule
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.strict = strict
        self.check_finite = check_finite
        self.wait_count = 0
        self.stopped_epoch = 0
        self.log_rank_zero_only = log_rank_zero_only

        if self.mode not in self.mode_dict:
            raise MisconfigurationException(f"`mode` can be {', '.join(self.mode_dict.keys())}, got {self.mode}")

        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf

    @property
    def state_key(self) -> str:
        return self._generate_state_key(monitor=self.monitor, mode=self.mode)

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        return

    def _validate_condition_metric(self, logs: Dict[str, Tensor]) -> bool:
        monitor_val = logs.get(self.monitor)

        error_msg = (
            f"Task canceling conditioned on metric `{self.monitor}` which is not available."
            " Pass in or modify your `MultimodalOverfittingMonitor` callback to use any of the following:"
            f' `{"`, `".join(list(logs.keys()))}`'
        )

        if monitor_val is None:
            if self.strict:
                raise RuntimeError(error_msg)
            if self.verbose > 0:
                rank_zero_warn(error_msg, category=RuntimeWarning)

            return False

        return True

    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "wait_count": self.wait_count,
            "stopped_epoch": self.stopped_epoch,
            "best_score": self.best_score,
            "patience": self.patience,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.wait_count = state_dict["wait_count"]
        self.stopped_epoch = state_dict["stopped_epoch"]
        self.best_score = state_dict["best_score"]
        self.patience = state_dict["patience"]

    def _should_skip_check(self, trainer: "pl.Trainer") -> bool:
        from pytorch_lightning.trainer.states import TrainerFn

        return trainer.state.fn != TrainerFn.FITTING or trainer.sanity_checking

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        return

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer, pl_module)

        if wandb.run is not None:  # https://github.com/wandb/wandb/issues/3551
            prefix = "validation/monitor"
            match self.monitor:
                case "validation/losses/itm_loss":
                    trainer.logger.experiment.log({f"{prefix}/itm": pl_module.model.itm_weight})
                case "validation/losses/global_contrastive_loss":
                    trainer.logger.experiment.log(
                        {f"{prefix}/global_contrastive": pl_module.model.global_contrastive_weight}
                    )
                case "validation/losses/mmm_image_loss":
                    trainer.logger.experiment.log({f"{prefix}/mmm_image": pl_module.model.mmm_image_weight})
                case "validation/losses/mmm_text_loss":
                    trainer.logger.experiment.log({f"{prefix}/mmm_text": pl_module.model.mmm_text_weight})
                case "validation/losses/mlm_loss":
                    weight = self.datamodule.sampling_weights[2 if len(self.datamodule.sampling_weights) > 1 else 0]
                    trainer.logger.experiment.log({f"{prefix}/mlm": weight})
                case "validation/losses/mim_loss":
                    weight = self.datamodule.sampling_weights[1 if len(self.datamodule.sampling_weights) > 1 else 0]
                    trainer.logger.experiment.log({f"{prefix}/mim": weight})

    def _run_early_stopping_check(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Checks whether the cancel task condition is met and if so tells the model loss to ignore the task."""
        logs = trainer.callback_metrics

        if trainer.fast_dev_run or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
                logs
        ):  # short circuit if metric not present
            return

        current = logs[self.monitor].squeeze()
        patience_exhausted, reason, wait_count_increased = self._evaluate_stopping_criteria(current)

        # slow down if any world process detects over-fitting
        wait_count_increased = trainer.strategy.reduce_boolean_decision(wait_count_increased, all=False)
        if wait_count_increased:
            match self.monitor:
                case "validation/losses/itm_loss":
                    pl_module.model.itm_weight /= 2.
                case "validation/losses/global_contrastive_loss":
                    pl_module.model.global_contrastive_weight /= 2.
                case "validation/losses/mmm_image_loss":
                    pl_module.model.mmm_image_weight /= 2.
                case "validation/losses/mmm_text_loss":
                    pl_module.model.mmm_text_weight /= 2.
                case "validation/losses/mlm_loss":
                    self._set_sampling_weight_for_modality("text", type="half")
                case "validation/losses/mim_loss":
                    self._set_sampling_weight_for_modality("vision", type="half")

        # stop every ddp process if any world process decides to stop
        patience_exhausted = trainer.strategy.reduce_boolean_decision(patience_exhausted, all=False)
        if patience_exhausted:
            match self.monitor:
                case "validation/losses/itm_loss":
                    pl_module.model.itm_weight = 0.
                case "validation/losses/global_contrastive_loss":
                    pl_module.model.global_contrastive_weight = 0.
                case "validation/losses/mmm_image_loss":
                    pl_module.model.mmm_image_weight = 0.
                case "validation/losses/mmm_text_loss":
                    pl_module.model.mmm_text_weight = 0.
                    if self.datamodule.sampling_weights[2] == 0.:  # With no MLM and MMM_text, stop training
                        self._stop_training(trainer)
                case "validation/losses/mlm_loss":
                    self._set_sampling_weight_for_modality("text", trainer=trainer, type="zero")
                    if pl_module.model.mmm_text_weight == 0.:  # With no MLM and MMM_text, stop training
                        self._stop_training(trainer)
                case "validation/losses/mim_loss":
                    self._set_sampling_weight_for_modality("vision", trainer=trainer, type="zero")

            multimodal_tasks_weights = [pl_module.model.itm_weight,
                                        pl_module.model.global_contrastive_weight,
                                        pl_module.model.mmm_image_weight,
                                        pl_module.model.mmm_text_weight]
            if all([w == 0. for w in multimodal_tasks_weights]):
                self._set_sampling_weight_for_modality("multimodal", trainer=trainer, type="zero")

        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)

    def _stop_training(self, trainer):
        trainer.should_stop = True
        self.stopped_epoch = trainer.current_epoch

    def _set_sampling_weight_for_modality(self, modality: str, trainer: "pl.Trainer" = None, type: str = "zero"):
        weights = self.datamodule.sampling_weights
        if len(weights) == 1.:
            if type == "zero":
                self._stop_training(trainer)
        else:
            if modality == "multimodal":
                new_value = weights[0] / 2. if type == "half" else 0.
                self.datamodule.update_sampling_function_and_weights([new_value, weights[1], weights[2]])
            elif modality == "text":
                new_value = weights[2] / 2. if type == "half" else 0.
                self.datamodule.update_sampling_function_and_weights([weights[0], weights[1], new_value])
            elif modality == "vision":
                new_value = weights[1] / 2. if type == "half" else 0.
                self.datamodule.update_sampling_function_and_weights([weights[0], new_value, weights[2]])
            else:
                raise ValueError(f"Modality {modality} not recognized.")

    def _evaluate_stopping_criteria(self, current: Tensor) -> Tuple[bool, Optional[str], bool]:
        should_stop = False
        reason = None
        wait_count_increased = False
        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}. Signaling model to ignore task."
            )
        elif self.monitor_op(current - self.min_delta, self.best_score.to(current.device)):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            wait_count_increased = True
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Signaling model to ignore task."
                )

        return should_stop, reason, wait_count_increased

    def _improvement_message(self, current: Tensor) -> str:
        """Formats a log message that informs the user about an improvement in the monitored score."""
        if torch.isfinite(self.best_score):
            msg = (
                f"Metric {self.monitor} improved by {abs(self.best_score - current):.3f} >="
                f" min_delta = {abs(self.min_delta)}. New best score: {current:.3f}"
            )
        else:
            msg = f"Metric {self.monitor} improved. New best score: {current:.3f}"
        return msg

    @staticmethod
    def _log_info(trainer: Optional["pl.Trainer"], message: str, log_rank_zero_only: bool) -> None:
        rank = _get_rank(
            strategy=(trainer.strategy if trainer is not None else None),  # type: ignore[arg-type]
        )
        if trainer is not None and trainer.world_size <= 1:
            rank = None
        message = rank_prefixed_message(message, rank)
        if rank is None or not log_rank_zero_only or rank == 0:
            log.info(message)
