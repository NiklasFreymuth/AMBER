import math
from typing import Union

import torch
import torch.nn as nn
import tqdm
from torch_geometric.data import HeteroData
from torch_geometric.data.data import Data

from src.hmpn.common.hmpn_util import make_batch


def estimate_max_batch_size(model: nn.Module, input_sample: Union[torch.Tensor, Data, HeteroData],
                            device: str = "cuda", rel_tolerance: float = 0.1, verbose: bool = False) -> int:
    """
    Estimate the maximum batch size for a given model and input sample, considering training (forward and backward pass)
    Assumes that all constant tensors such as model weights are already on the device.
    Will run a binary search to find the maximum batch size that fits into the GPU without overflowing.
    This provokes several overflows, but those should be handled by torch.cuda.empty_cache().

    Args:
    - model: The PyTorch model to be tested.
    - input_sample: A sample input tensor. The shape should be (1, ...), or a single pyg graph object.
    - device: Device to run the model on.
    - rel_tolerance: Relative tolerance for the batch size.  Given max_batch_size, the function will return
        the largest batch size that is at most (1 + rel_tolerance) * max_batch_size.

    Returns:
    - int: Maximum batch size that fits into the GPU without overflowing.
    """
    batch_size = 1
    success_batch_size = 0

    # Prepare model
    model.to(device)
    model.train()

    input_sample = input_sample.to(device)

    if verbose:
        print("Estimating maximum batch size...")

    # Run a binary search to find the first batch size that overflows
    for _ in tqdm.tqdm(range(100), disable=not verbose, desc="Initial search"):
        try:
            _model_call(model, input_sample, batch_size)
            success_batch_size = batch_size
            batch_size *= 2  # Double the batch size for the next trial

        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                break
            else:
                raise e

    # Search between success_batch_size and batch_size
    max_iters = int(math.log2(success_batch_size)) + 1  # maximum iterations for binary search
    for _ in tqdm.tqdm(range(max_iters), disable=not verbose, desc="Binary search"):
        mid_batch_size = (success_batch_size + batch_size) // 2
        try:
            _model_call(model, input_sample, mid_batch_size)
            success_batch_size = mid_batch_size
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                batch_size = mid_batch_size
            else:
                raise e

    max_batch_size = int(success_batch_size * (1 - rel_tolerance))
    if verbose:
        print(f"Estimated maximum batch size, including tolerance of {rel_tolerance}: {max_batch_size}")

    # sync and clear memory
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    return max_batch_size


def _model_call(model: nn.Module, input_sample: Union[torch.Tensor, Data, HeteroData], batch_size: int) -> None:
    """
    Call the model with a given batch size and run a backward pass to estimate the memory usage.
    Args:
        model: The model to be tested.
        input_sample: A single sample input tensor or a single pyg graph object.
        batch_size: The batch size to test. Will repeat the input sample batch_size times to create a batch.

    Returns:

    """
    # clear up memory for accurate estimate
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # Create a batch of input samples
    if isinstance(input_sample, (Data, HeteroData)):
        graphs = [input_sample] * batch_size
        input_batch = make_batch(graphs)
    else:
        input_batch = input_sample.repeat(batch_size, *([1] * (input_sample.dim() - 1)))

    output = model(input_batch)

    if isinstance(output, tuple):
        node_output = output[0][list(output[0].keys())[0]]
    elif isinstance(output, torch.Tensor):
        node_output = output
    else:
        raise ValueError(f"Unsupported output type {type(output)}")
    node_output.sum().backward()
