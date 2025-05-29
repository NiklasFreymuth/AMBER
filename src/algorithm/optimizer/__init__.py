from typing import Any, Dict, Union

import torch
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.algorithm.optimizer.learning_rate_schedulers import get_lr_scheduler
from src.algorithm.optimizer.optimizers import get_optimizer


def get_optimizer_and_scheduler(
    optimizer_dict: DictConfig, model: torch.nn.Module, num_epochs: int
) -> Dict[str, Union[Optimizer, Dict[str, Union[LRScheduler, str]]]]:
    """
    Returns a dictionary containing both the optimizer and, if applicable, the learning rate scheduler.

    Args:
        optimizer_dict (Dict): A dictionary containing:
            - "optimizer_type" (str): The type of optimizer to use.
            - "scheduler_type" (str): The type of learning rate scheduler to use.
            - "lr" (float): The learning rate for the optimizer.
        model (torch.nn.Module): The model whose parameters will be optimized.
        num_epochs (int): The total number of training epochs.

    Returns:
        Dict[str, Union[Optimizer, Dict[str, Union[_LRScheduler, str]]]]]: A dictionary containing the optimizer and scheduler.

    Raises:
        ValueError: If an unsupported optimizer or scheduler type is provided.
    """
    optimizer_type = optimizer_dict.name
    lr = optimizer_dict.lr
    weight_decay = optimizer_dict.weight_decay
    optimizer = get_optimizer(model, optimizer_type=optimizer_type, lr=lr, weight_decay=weight_decay)

    scheduler_type = optimizer_dict.get("scheduler", None)
    scheduler_config = get_lr_scheduler(optimizer, scheduler_type=scheduler_type, num_epochs=num_epochs)

    if scheduler_config:
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
    else:
        return {"optimizer": optimizer}
