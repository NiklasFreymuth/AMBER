from dataclasses import dataclass
from typing import Dict

from lightning import LightningModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.algorithm import create_algorithm
from src.algorithm.core.amber import Amber
from src.algorithm.dataloader import get_dataloader, get_datasets
from src.initialization.init_config import initialize_config
from src.initialization.init_seed import initialize_seed
from src.logger import CustomWandBLogger, get_wandb_logger


@dataclass
class InitializationReturn:
    """
    Data structure to store the results of the initialization process.

    Attributes:
        dataloaders (Dict[str, DataLoader]): A dictionary mapping dataset modes (e.g., "train", "val")
            to their corresponding PyTorch DataLoader instances.
        algorithm (Amber): The instantiated algorithm object based on the provided configuration.
        wandb_logger (Union[bool, CustomWandBLogger]): If Weights & Biases (WandB) logging is enabled,
            this will be an instance of CustomWandBLogger; otherwise, it will be False.
    """

    dataloaders: Dict[str, DataLoader]
    algorithm: LightningModule
    wandb_logger: bool | CustomWandBLogger


def initialize(*, config: DictConfig) -> InitializationReturn:
    """
    Initializes the training setup by setting up configurations, loading datasets,
    creating dataloaders, instantiating the algorithm, and configuring logging.

    Args:
        config (DictConfig): Configuration object containing experiment settings.

    Returns:
        InitializationReturn: A data structure containing the initialized dataloaders, algorithm,
        and WandB logger (if enabled).
    """
    initialize_config(config)  # Apply SLURM-related settings to the configuration
    initialize_seed(config.seed)  # Set random seed for reproducibility

    datasets = get_datasets(algorithm_config=config.algorithm, task_config=config.task)

    # Create dataloaders for each dataset mode (e.g., "train", "val", "test)
    dataloaders = {
        dataset_mode: get_dataloader(algorithm_config=config.algorithm, dataset=dataset, is_train=dataset_mode == "train")
        for dataset_mode, dataset in datasets.items()
    }

    # Instantiate the algorithm using the provided configuration
    algorithm = create_algorithm(algorithm_config=config.algorithm, train_dataset=datasets.get("train"))

    # Initialize WandB logger if enabled in the configuration
    if config.logger.wandb.enabled:
        wandb_logger = get_wandb_logger(config=config)
    else:
        wandb_logger = False
    return InitializationReturn(dataloaders, algorithm, wandb_logger)
