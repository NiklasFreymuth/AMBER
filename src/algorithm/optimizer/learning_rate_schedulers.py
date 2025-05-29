from typing import Callable, Dict, Optional, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def get_lr_scheduler(optimizer: Optimizer, scheduler_type: str | None, num_epochs: int) -> Optional[Dict[str, Union[LRScheduler, str]]]:
    """
    Returns the appropriate learning rate scheduler based on the given type.

    Args:
        optimizer (Optimizer): The optimizer to which the scheduler will be applied.
        scheduler_type (str): The type of scheduler to use. Options: "linear_warmup_decay", "none".
        num_epochs (int): The total number of epochs (only used for scheduling).

    Returns:
        Optional[Dict[str, Union[_LRScheduler, str]]]: The scheduler dictionary compatible with PyTorch Lightning,
        or None if no scheduling is used.
    """
    if scheduler_type == "linear_warmup_decay":
        scheduler = LambdaLR(optimizer, lr_lambda=linear_warmup_decay(num_epochs))
        return {"scheduler": scheduler, "interval": "epoch"}
    elif scheduler_type is None or scheduler_type is False:
        return None  # No learning rate scheduling

    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


def linear_warmup_decay(num_epochs: int, warmup_ratio: float = 0.1) -> Callable[[int], float]:
    """
    Returns a lambda function for linear warmup followed by linear decay.

    Args:
        num_epochs (int): Total number of epochs.
        warmup_ratio (float): Fraction of total epochs used for warmup. Defaults to 0.1.

    Returns:
        Callable[[int], float]: A function that computes the learning rate scaling factor.
    """
    warmup_steps = int(num_epochs * warmup_ratio)
    total_steps = num_epochs

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / warmup_steps  # Linear warmup (0 to 1)
        return max(0.0, (total_steps - current_step) / (total_steps - warmup_steps))  # Linear decay (1 to 0)

    return lr_lambda
