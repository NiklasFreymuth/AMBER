from typing import Dict, List, Union

import torch
from numpy import ndarray
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData


def detach(tensor: Union[Tensor, Dict[str, Tensor], List[Tensor]]) -> Union[ndarray, Dict[str, ndarray], List[ndarray], BaseData]:
    if isinstance(tensor, dict):
        return {key: detach(value) for key, value in tensor.items()}
    elif isinstance(tensor, list):
        return [detach(value) for value in tensor]
    if tensor.is_cuda:
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()


def count_parameters(model):
    """
    Counts the total number of parameters in a PyTorch model.

    Args:
    model (nn.Module): The model whose parameters are to be counted.

    Returns:
    int: The total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch._dynamo.disable
def make_batch(data: torch.Tensor | Batch | Data | List[Data] | List[torch.Tensor], **kwargs):
    """
    adds the .batch-argument with zeros
    Args:
        data:

    Returns:

    """
    if isinstance(data, Data):
        return Batch.from_data_list([data], **kwargs)
    elif isinstance(data, Batch):
        return data
    elif isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, list) and isinstance(data[0], Data):
        return Batch.from_data_list(data, **kwargs)
    elif isinstance(data, list) and isinstance(data[0], torch.Tensor):
        return torch.cat(data, dim=0)
    else:
        raise ValueError(f"Unsupported data type {type(data)}")


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))
