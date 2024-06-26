import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from util.types import *


def detach(
    tensor: Union[Tensor, Dict[Key, Tensor], List[Tensor]]
) -> Union[ndarray, Dict[Key, ndarray], List[ndarray], BaseData]:
    if isinstance(tensor, dict):
        return {key: detach(value) for key, value in tensor.items()}
    elif isinstance(tensor, list):
        return [detach(value) for value in tensor]
    if tensor.is_cuda:
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()


def normalize(tensor: Tensor, epsilon: float = 1.0e-8, dim: int = 0) -> Tensor:
    return (tensor - torch.mean(tensor, dim=dim)) / (torch.std(tensor, dim=dim) + epsilon)


def assert_same_shape(reference_tensor: Tensor, *other_tensors):
    for other_tensor in other_tensors:
        assert reference_tensor.shape == other_tensor.shape


def get_optimizer(optimizer_name: str) -> Type[Union[Adam, AdamW, SGD]]:
    assert optimizer_name is not None, "Need to select an optimizer from ['adam', 'sgd']. Given None instead."
    if optimizer_name.lower() == "adam":
        return Adam
    elif optimizer_name.lower() == "adamw":
        return AdamW
    elif optimizer_name.lower() == "sgd":
        return SGD
    else:
        raise ValueError(f"Unknown optimizer '{optimizer_name}'")


def orthogonal_initialization(module: nn.Module, gain: float = 2**0.5) -> None:
    """
    Performs an orthogonal initialization for the parameters of the given module in-place.
    Args:
        module: Module containing layers to initialize orthogonally
        gain: Scale of the initialization

    Returns: Nothing, as the initialization is done in-place

    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


def count_parameters(model):
    """
    Counts the total number of parameters in a PyTorch model.

    Args:
    model (nn.Module): The model whose parameters are to be counted.

    Returns:
    int: The total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
