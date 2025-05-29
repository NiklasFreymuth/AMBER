import copy
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

from numpy import ndarray
from torch import Tensor
from torch_geometric.data.batch import Batch
from torch_geometric.data.data import BaseData, Data
from torch_geometric.data.hetero_data import HeteroData

"""
Custom class that redefines various types to increase clarity.
"""
Key = Union[str, int]  # for dictionaries, we usually use strings or ints as keys
ConfigDict = Dict[Key, Any]  # A (potentially nested) dictionary containing the "params" section of the .yaml file
EntityDict = Dict[Key, Union[Dict, str]]  # potentially nested dictionary of entities
ValueDict = Dict[Key, Any]
Result = Union[List, int, float, ndarray]
Shape = Union[int, Iterable, ndarray]

InputBatch = Union[Dict[Key, Tensor], Tensor, Batch, Data, HeteroData, None]
OutputTensorDict = Dict[Key, Tensor]
