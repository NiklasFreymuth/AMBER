import os
from argparse import Namespace
from typing import Any, Dict, Optional, Union

from lightning.pytorch.loggers import Logger

from src.logger.evaluation_logger_experiment import EvaluationLoggerExperiment


class EvaluationLogger(Logger):
    def __init__(self, output_path, exp_name, job_type, seed, save_figures: bool):
        super().__init__()
        self.output_path = str(os.path.join(output_path, exp_name, job_type, seed))
        self._vis_path = str(os.path.join(self.output_path, "visualizations"))
        # create output path
        os.makedirs(self.output_path, exist_ok=True)
        self.experiment = EvaluationLoggerExperiment(output_path=self.output_path, save_figures=save_figures)

    @property
    def name(self) -> Optional[str]:
        return "Test Logger"

    @property
    def version(self) -> Optional[Union[int, str]]:
        return "1.0"

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace], *args: Any, **kwargs: Any) -> None:
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        pass

    def save(self):
        # Optionally implement saving or finalizing logging state if needed
        pass

    def finalize(self, status):
        # Finalize logging when done (e.g., close connections)
        pass
