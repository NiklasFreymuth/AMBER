from typing import List

from src.algorithms.abstract_iterative_algorithm import AbstractIterativeAlgorithm
from src.recording.loggers.abstract_logger import AbstractLogger
from util.types import ConfigDict


def create_loggers(config: ConfigDict, algorithm: AbstractIterativeAlgorithm) -> List[AbstractLogger]:
    """
    Create a list of all loggers used for the current run. The order of the loggers may matter, since loggers can pass
    computed values to subsequent ones.
    Args:
        config: A (potentially nested) dictionary containing the "params" section of the section in the .yaml file
            used by cw2 for the current run.
        algorithm: An instance of the algorithm to run.

    Returns: A list of loggers to use.

    """
    from src.recording.loggers.config_logger import ConfigLogger
    from src.recording.loggers.scalars_logger import ScalarsLogger
    from util.function import get_from_nested_dict

    recording_dict = config.get("recording", {})

    logger_classes = [ConfigLogger, ScalarsLogger]
    if get_from_nested_dict(recording_dict, list_of_keys=["wandb", "enabled"], default_return=False):
        from src.recording.loggers.custom_wandb_logger import CustomWAndBLogger

        logger_classes.append(CustomWAndBLogger)
    if recording_dict.get("checkpoint"):
        from src.recording.loggers.checkpoint_logger import CheckpointLogger

        logger_classes.append(CheckpointLogger)
    if recording_dict.get("end_of_training_evaluation"):
        from src.recording.loggers.end_of_training_logger import EndOfTrainingLogger

        logger_classes.append(EndOfTrainingLogger)

    loggers = [logger(config=config, algorithm=algorithm) for logger in logger_classes]
    return loggers
