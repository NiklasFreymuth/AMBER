from typing import Dict, Union, Optional, Tuple, Any

from torch_geometric.data.data import Data

from src.hmpn.abstract.abstract_message_passing_base import AbstractMessagePassingBase

def get_hmpn_from_graph(*, example_graph: Data,
                        latent_dimension: int,
                        base_config: Dict[str, Any],
                        node_name: str = "node",
                        unpack_output: bool = True,
                        device: Optional = None) -> AbstractMessagePassingBase:
    """
    Build and return a Message Passing Base specified in the config from the provided example graph.
    Args:
        example_graph: A graph that is used to infer the input feature dimensions for nodes, edges
        latent_dimension: Dimensionality of the latent space
        base_config: Dictionary specifying the way that the gnn base should
        node_name: Name of the node for homogeneous graphs. Will be used for the unpacked output namingook like.
        unpack_output: If true, will unpack the processed batch of graphs to a 4-tuple of
            ({node_name: node features}, {edge_name: edge features}, {node_name: batch indices}).
            Else, will return the raw processed batch of graphs
        device: The device to put the base on. Either cpu or a single gpu

    Returns:

    """
    in_node_features = example_graph.x.shape[1]
    in_edge_features = example_graph.edge_attr.shape[1]
    return get_hmpn(in_node_features=in_node_features,
                    in_edge_features=in_edge_features,
                    latent_dimension=latent_dimension,
                    base_config=base_config,
                    node_name=node_name,
                    unpack_output=unpack_output,
                    device=device)


def get_hmpn(*,
             in_node_features: int,
             in_edge_features: int,
             latent_dimension: int,
             base_config: Dict[str, Any],
             node_name: str = "node",
             unpack_output: bool = True,
             device: Optional = None) -> AbstractMessagePassingBase:
    """
    Build and return a Message Passing Base specified in the config.

    Args:
        in_node_features:
        in_edge_features:
        latent_dimension: The dimension of the latent space
        base_config: The config for the base
        node_name: Name of the node for homogeneous graphs. Will be used for the unpacked output naming
        unpack_output: If true, will unpack the processed batch of graphs to a 4-tuple of
            ({node_name: node features}, {edge_name: edge features}, {node_name: batch indices}).
            Else, will return the raw processed batch of graphs
        device: The device to put the base on. Either cpu or a single gpu

    Returns:

    """
    assert type(in_node_features) == type(in_edge_features), f"May either provide feature dimensions as int or Dict, " \
                                                             f"but not both. " \
                                                             f"Given '{in_node_features}', '{in_edge_features}'"

    create_graph_copy = base_config.get("create_graph_copy")
    assert_graph_shapes = base_config.get("assert_graph_shapes")
    stack_config = base_config.get("stack")
    embedding_config = base_config.get("embedding")
    scatter_reduce_strs = base_config.get("scatter_reduce")
    flip_edges_for_nodes = base_config.get('flip_edges_for_nodes', False)
    edge_dropout = base_config.get('edge_dropout', 0.0)
    if isinstance(scatter_reduce_strs, str):
        scatter_reduce_strs = [scatter_reduce_strs]

    params = dict(in_node_features=in_node_features,
                  in_edge_features=in_edge_features,
                  latent_dimension=latent_dimension,
                  scatter_reduce_strs=scatter_reduce_strs,
                  stack_config=stack_config,
                  unpack_output=unpack_output,
                  embedding_config=embedding_config,
                  flip_edges_for_nodes=flip_edges_for_nodes,
                  edge_dropout=edge_dropout,
                  create_graph_copy=create_graph_copy,
                  assert_graph_shapes=assert_graph_shapes,
                  )
    from src.hmpn.homogeneous.homogeneous_message_passing_base import HomogeneousMessagePassingBase
    base = HomogeneousMessagePassingBase(**params,
                                         node_name=node_name)
    base = base.to(device)
    return base
