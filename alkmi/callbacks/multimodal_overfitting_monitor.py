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

MMM_TEXT_THRESHOLD = 0.06  # insignificant enough that we'd like to stop training


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
        >>> from alkmi.callbacks.multimodal_overfitting_monitor import MultimodalOverfittingMonitor
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
            original_weight: float,
            retry_patience: int = 8,
            min_delta: float = 0.0,
            patience: int = 3,
            verbose: bool = True,
            mode: str = "min",
            strict: bool = True,
            check_finite: bool = True,
            log_rank_zero_only: bool = False,
    ):
        super().__init__()
        self.monitor = monitor
        self.name = self.monitor.split('validation/losses/')[-1].split('_loss')[0]
        assert self.name in ["mlm", "mim", "itm", "global_contrastive", "mmm_image", "mmm_text"], \
            f"Monitor {self.monitor} not currently supported. Please consider adding support for it."
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

        self.original_weight = original_weight
        self.retry_patience = retry_patience
        assert retry_patience > patience, "Retry patience must be greater than the patience (must wait for a bit)."

        if self.mode not in self.mode_dict:
            raise MisconfigurationException(f"`mode` can be {', '.join(self.mode_dict.keys())}, got {self.mode}")

        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf

    def _get_weight(self, pl_module):
        weights = self.datamodule.sampling_weights  # [mmm, mim, mlm]
        if self.name == "mlm":
            return weights[2 if len(weights) > 1 else 0]
        elif self.name == "mim":
            return weights[1 if len(weights) > 1 else 0]
        else:
            return pl_module.model.__getattribute__(f"{self.name}_weight")

    def _set_weight(self, trainer, pl_module, new_weight: float):
        # For debugging purposes:
        # print(f"Setting {self.name}_weight to {new_weight} (was {self._get_weight(pl_module)})")

        is_flava_model = hasattr(pl_module.model, 'mmm_text_weight')
        weights = self.datamodule.sampling_weights

        if len(weights) == 1. and new_weight == 0.:
            self._stop_training(trainer)
        elif self.name == "mlm":
            self.datamodule.update_sampling_function_and_weights([weights[0], weights[1], new_weight])
            if new_weight == 0. and is_flava_model and pl_module.model.mmm_text_weight < MMM_TEXT_THRESHOLD:
                self._stop_training(trainer)  # With no MLM and MMM_text, stop training
        elif self.name == "mim":
            self.datamodule.update_sampling_function_and_weights([weights[0], new_weight, weights[2]])
        else:
            pl_module.model.__setattr__(f"{self.name}_weight", new_weight)
            if new_weight == 0. and self.name == "mmm_text" and weights[2] == 0.:
                self._stop_training(trainer)  # With no MLM and MMM_text, stop training

        if is_flava_model:
            multimodal_tasks_weights = ["itm", "global_contrastive", "mmm_image", "mmm_text"]
            if all([pl_module.model.__getattribute__(f"{w}_weight") == 0. for w in multimodal_tasks_weights]):
                self.datamodule.update_sampling_function_and_weights([0., weights[1], weights[2]])

        if all([x == 0. for x in weights]):
            self._stop_training(trainer)

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
        if self.name in ['mlm', 'mmm_text']:
            print(f"Resetting wait count for `{self.name}` since training is resuming and they are special tasks.")
            self.wait_count = 0
        else:
            self.wait_count = state_dict["wait_count"]
        self.stopped_epoch = state_dict["stopped_epoch"]
        self.best_score = state_dict["best_score"]
        # Patience values might change across runs
        # self.patience = state_dict["patience"]

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
            trainer.logger.experiment.log({f"validation/monitor/{self.name}": self._get_weight(pl_module)})

    def _run_early_stopping_check(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Checks whether the cancel task condition is met and if so tells the model loss to ignore the task."""
        logs = trainer.callback_metrics

        if trainer.fast_dev_run or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
                logs
        ):  # short circuit if metric not present
            return

        reason = None
        if self.wait_count >= self.retry_patience:
            reason = (
                f"Monitored metric {self.monitor} did not improve in {self.wait_count} records."
                f" This exceeds the restart patience {self.retry_patience}, so we will restart the task with"
                f" renewed patience."
            )
            self._set_weight(trainer, pl_module, 0.1 * self.original_weight)
            self.wait_count = 0
        else:
            current = logs[self.monitor].squeeze()
            patience_exhausted, reason, wait_count_increased = self._evaluate_stopping_criteria(current)

            # slow down if any world process detects over-fitting
            wait_count_increased = trainer.strategy.reduce_boolean_decision(wait_count_increased, all=False)
            if wait_count_increased:
                # stop every ddp process if any world process decides to stop
                patience_exhausted = trainer.strategy.reduce_boolean_decision(patience_exhausted, all=False)
                if patience_exhausted:
                    self._set_weight(trainer, pl_module, 0.)  # stop task
                else:
                    halved_weight = self._get_weight(pl_module) / 2.
                    self._set_weight(trainer, pl_module, halved_weight)
            else:
                increased_weight = min(self.original_weight, self._get_weight(pl_module) * 1.1)
                self._set_weight(trainer, pl_module, increased_weight)

        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)

    def _stop_training(self, trainer):
        trainer.should_stop = True
        self.stopped_epoch = trainer.current_epoch

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
            else:
                reason = (
                    f"Monitored metric {self.monitor} did not improve since the last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Wait count has been increased from {self.wait_count - 1}."
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
            print(message)
