from typing import Any, Dict, Optional, Tuple, Union

from torch_geometric.data.data import Data

from src.mpn.message_passing_base import MessagePassingBase


def get_mpn_from_graph(
    *,
    example_graph: Data,
    latent_dimension: int,
    base_config: Dict[str, Any],
    node_name: str = "node",
    device: Optional = None,
) -> MessagePassingBase:
    """
    Build and return a Message Passing Base specified in the config from the provided example graph.
    Returns:

    """
    in_node_features = example_graph.x.shape[1]
    in_edge_features = example_graph.edge_attr.shape[1]
    return get_mpn(
        in_node_features=in_node_features,
        in_edge_features=in_edge_features,
        latent_dimension=latent_dimension,
        base_config=base_config,
        node_name=node_name,
        device=device,
    )


def get_mpn(
    *,
    in_node_features: Union[int, Dict[str, int]],
    in_edge_features: Union[int, Dict[Tuple[str, str, str], int]],
    latent_dimension: int,
    base_config: Dict[str, Any],
    node_name: str = "node",
    device: Optional = None,
) -> MessagePassingBase:
    """
    Build and return a Message Passing Base specified in the config.
    """
    assert type(in_node_features) == type(in_edge_features), (
        f"May either provide feature dimensions as int or Dict, " f"but not both. " f"Given '{in_node_features}', '{in_edge_features}'"
    )

    create_graph_copy = base_config.get("create_graph_copy")
    assert_graph_shapes = base_config.get("assert_graph_shapes")
    stack_config = base_config.get("stack")
    embedding_config = base_config.get("embedding")
    edge_dropout = base_config.get("edge_dropout")

    params = dict(
        in_node_features=in_node_features,
        in_edge_features=in_edge_features,
        latent_dimension=latent_dimension,
        stack_config=stack_config,
        embedding_config=embedding_config,
        edge_dropout=edge_dropout,
        create_graph_copy=create_graph_copy,
        assert_graph_shapes=assert_graph_shapes,
    )

    base = MessagePassingBase(**params, node_type=node_name)
    base = base.to(device)
    return base
