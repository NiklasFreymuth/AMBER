import os
from typing import Mapping

from lightning_fabric.utilities.logger import _add_prefix
from lightning_utilities.core.imports import RequirementCache
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only

_WANDB_AVAILABLE = RequirementCache("wandb>=0.12.10")


def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
        "WANDB_START_METHOD",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


class CustomWandBLogger(WandbLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    @rank_zero_only
    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: int,
    ) -> None:
        # This is exactly like the regular WandbLogger, except the step must be
        # specified, and images are logged correctly according to the step.
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        if "epoch" in metrics:
            assert metrics["epoch"] >= self._epoch, "epoch must not decrease"
            self._epoch = metrics["epoch"]
        # assert "epoch" in metrics, "epoch must be in the metrics"
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)

        self.experiment.log(dict(metrics, **{"trainer/global_step": step}), step=self.epoch)
