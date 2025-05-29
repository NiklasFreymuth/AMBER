from functools import partial, update_wrapper
from typing import Any, Dict, Iterable, List, Union

import numpy as np

# A nested configuration where leaf dictionaries map strings to booleans.
NestedConfig = Dict[str, Union[bool, "NestedConfig"]]


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


class classproperty(property):
    def __get__(self, instance, owner):
        return self.fget(owner)


def filter_included_fields(config: NestedConfig) -> Union[List[str], Dict[str, Union[List[str], "NestedConfig"]]]:
    """
    Recursively transforms a nested configuration dictionary into a structure where
    each leaf dictionary (i.e. one mapping strings to booleans) is replaced by a list of keys
    whose values are True.

    Args:
        config: A nested dictionary where at the leaf level, each value is a boolean.

    Returns:
        If the current dictionary is a leaf (all values are booleans), returns a list of keys with True.
        Otherwise, returns a dictionary with the same keys, where each value is processed recursively.
    """
    if all(isinstance(v, bool) for v in config.values()):
        # Leaf level: filter keys with True values.
        return [k for k, v in config.items() if v]
    else:
        # Recursive step: process each nested dictionary.
        return {k: filter_included_fields(v) for k, v in config.items()}


def prefix_keys(dictionary: Dict[str, Any], prefix: str | List[str], separator: str = "/") -> Dict[str, Any]:
    if isinstance(prefix, str):
        prefix = [prefix]
    prefix = separator.join(prefix + [""])
    return {prefix + k: v for k, v in dictionary.items()}


def add_to_dictionary(dictionary: Dict, new_scalars: Dict) -> Dict:
    for k, v in new_scalars.items():
        if k not in dictionary:
            dictionary[k] = []
        if isinstance(v, list) or (isinstance(v, np.ndarray) and v.ndim == 1):
            dictionary[k].extend(list(v))
        else:
            dictionary[k].append(v)
    return dictionary


def safe_mean(arr: np.ndarray | list) -> np.ndarray:
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.
    """
    return np.nan if len(arr) == 0 else np.mean(arr)


def round_up_to_next(value: float):
    """
    Rounds the number to the next "1", "2", or "5"
    Args:
        value: Some float value. If positive, round up. If negative, round down

    Returns:

    """
    import math

    if value == 0:
        return 0
    elif value < 0:
        return round_up_to_next(-value)

    magnitude = 10 ** math.floor(math.log10(value))  # Get magnitude of value
    if value <= 2 * magnitude:
        return 2 * magnitude
    elif value <= 5 * magnitude:
        return 5 * magnitude
    else:
        return 10 * magnitude


def safe_concatenate(arrays: Iterable[np.array], *args, **kwargs) -> np.ndarray | None:
    """
    Concatenate arrays, but ignore None arrays
    Args:
        arrays:
        *args:
        **kwargs:

    Returns:

    """
    arrays = [array for array in arrays if array is not None]
    if len(arrays) == 0:
        return None
    return np.concatenate(arrays, *args, **kwargs)


def aggregate_metrics(metrics: List[Dict[str, float]]):
    """
    Aggregate metrics from list of dicts to a scalar per key:
    """
    epoch_averages: Dict[str, float] = {key: np.mean([output[key] for output in metrics if key in output]) for key in metrics[0].keys()}
    return epoch_averages
