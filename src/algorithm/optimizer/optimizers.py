import torch
from torch.optim import Optimizer


def get_optimizer(model: torch.nn.Module, optimizer_type: str, lr: float, weight_decay: float) -> Optimizer:
    """
    Returns the appropriate optimizer based on the given type.

    Args:
        model (torch.nn.Module): The model whose parameters will be optimized.
        optimizer_type (str): The type of optimizer to use. Supports "adam".
        lr (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay for the optimizer.

    Returns:
        Optimizer: The initialized optimizer.

    Raises:
        ValueError: If an unsupported optimizer type is provided.
    """
    if optimizer_type == "adam":
        from torch.optim import Adam

        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
