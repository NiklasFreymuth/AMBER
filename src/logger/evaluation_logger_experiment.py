import os
import re
from typing import Dict

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


def sanitize_filename(name, replace_with="_"):
    """Sanitize filenames by replacing invalid characters."""
    name = re.sub(r'[<>:"/\\|?*]', replace_with, name)  # Remove invalid chars
    name = re.sub(r"\s+", "_", name)  # Replace spaces with underscores
    name = name.strip("_")  # Remove leading/trailing underscores
    return name


class EvaluationLoggerExperiment:
    def __init__(self, output_path: str, save_figures: bool):
        self._output_path = output_path
        self._suffix = None
        self._save_figures = save_figures

    @property
    def output_path(self) -> str:
        if self._suffix:
            return os.path.join(self._output_path, self._suffix)
        return self._output_path

    @property
    def suffix(self) -> str:
        return self._suffix

    @suffix.setter
    def suffix(self, value: str):
        if not isinstance(value, str):
            raise ValueError("Suffix must be a string.")
        if not value:
            raise ValueError("Suffix cannot be empty.")
        self._suffix = value
        # Create the output directory if it doesn't exist
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

    def log(self, logging_dict: Dict, **kwargs):
        """
        Log the evaluation metrics to the output path. Go over the logging_dict, sort by type of logged value, and
            * save the figures as .html files
            * save the DataFrames as .csv files
            * save the scalars in a json file
        Args:
            logging_dict:
            **kwargs:

        Returns:

        """
        metrics = {}
        for key, value in logging_dict.items():
            if key.startswith("figure"):
                if self._save_figures:
                    # Save figure as .html file
                    file_name = sanitize_filename(key) + ".html"
                    value.write_html(os.path.join(self.output_path, file_name))
            elif isinstance(value, pd.DataFrame):
                # Save DataFrame as .csv file
                file_name = sanitize_filename(key) + ".csv"
                value.to_csv(os.path.join(self.output_path, file_name), index=False)
            else:
                # log (scalar) metric in json file
                if isinstance(value, (np.integer, np.floating)):
                    value = value.item()  # Convert to native Python type
                metrics[key] = value

        # log all metrics in json at output_path. Append in case this is called multiple times.
        with open(os.path.join(self.output_path, "evaluation_metrics.yaml"), "a") as f:
            f.write(OmegaConf.to_yaml(metrics, resolve=True))
