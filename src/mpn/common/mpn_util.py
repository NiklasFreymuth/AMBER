import copy
from typing import List, Union

import torch
import torch_geometric
from torch_geometric.data import Batch, Data


def noop(*args, **kwargs):
    """
    No-op function.
    Args:
        *args: Arguments to be passed to the function
        **kwargs: Keyword arguments to be passed to the function

    Returns: None

    """
    return None


def unpack_features(graph: Data, agent_node_type: str = "node"):
    """
    Unpacking important data from graphs.
    Args:
        graph (): The input observation
        agent_node_type: The name of the type of graph node that acts as the agent
     Returns:
        Tuple of edge_features, edge_index, node_features, and batch
    """
    # edge features
    edge_features = graph.edge_attr
    edge_index = graph.edge_index.long()  # cast to long for scatter operators

    # node features
    node_features = graph.x if graph.x is not None else graph.pos

    batch = graph.batch if hasattr(graph, "batch") else None
    if batch is None:
        batch = torch.zeros(node_features.shape[0]).long()

    return (
        {agent_node_type: node_features},
        {"edges": {"edge_index": edge_index, "edge_attr": edge_features}},
        {agent_node_type: batch},
    )


def make_batch(
    data: Union[
        Data,
        List[torch.Tensor],
        List[Data],
    ],
    **kwargs
):
    """
    adds the .batch-argument with zeros
    Args:
        data:

    Returns:

    """
    if isinstance(data, torch_geometric.data.Data):
        return Batch.from_data_list([data], **kwargs)
    elif isinstance(data, list) and isinstance(data[0], torch_geometric.data.Data):
        return Batch.from_data_list(data, **kwargs)
    elif isinstance(data, list) and isinstance(data[0], torch.Tensor):
        return torch.cat(data, dim=0)

    return data


def get_create_copy(create_graph_copy: bool):
    """
    Returns a function that creates a copy of the graph.
    Args:
        create_graph_copy: Whether to create a copy of the graph or not
    Returns: A function that creates a copy of the graph, or an empty function if create_graph_copy is False
    """
    if create_graph_copy:
        return lambda x: copy.deepcopy(x)
    else:
        return lambda x: x
