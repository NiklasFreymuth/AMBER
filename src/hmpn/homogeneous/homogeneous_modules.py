import abc
from typing import Dict, Any, List, Callable, Optional

import torch
from torch_geometric.data.batch import Batch

from src.hmpn.abstract.abstract_modules import AbstractMetaModule
from src.hmpn.common.latent_mlp import LatentMLP


class HomogeneousMetaModule(AbstractMetaModule, abc.ABC):
    """
    Base class for the homogeneous modules used in the GNN.
    They are used for updating node-, edge features.
    """

    def __init__(self, *,
                 in_features: int,
                 latent_dimension: int,
                 mlp_config: Optional[Dict[str, Any]],
                 scatter_reducers: List[Callable],
                 create_mlp: bool = True):
        """
        Args:
            in_features: Number of input features
            latent_dimension: Dimensionality of the internal layers of the mlp
            mlp_config: Dictionary specifying the way that the MLPs for each update should look like
            scatter_reducers: How to aggregate over the nodes/edges. Can for example be [torch.scatter_mean]
            create_mlp: Whether to create an MLP or not
        """
        super().__init__(scatter_reducers=scatter_reducers)
        if create_mlp:
            self._mlp = LatentMLP(in_features=in_features,
                                  latent_dimension=latent_dimension,
                                  config=mlp_config)
        else:
            self._mlp = None

    @property
    def mlp(self):
        assert self._mlp is not None, "MLP is not initialized"
        return self._mlp


class HomogeneousEdgeModule(HomogeneousMetaModule):
    """
    Module for computing edge updates of a step on a homogeneous message passing GNN. Edge inputs are concatenated:
    Its own edge features, the features of the two participating nodes and optionally,
    """

    def __init__(self, *,
                 latent_dimension: int,
                 mlp_config: Dict[str, Any],
                 scatter_reducers: List[Callable]):
        in_features = 3 * latent_dimension  # edge features, and the two participating nodes
        super(HomogeneousEdgeModule, self).__init__(in_features=in_features,
                                                    latent_dimension=latent_dimension,
                                                    mlp_config=mlp_config,
                                                    scatter_reducers=scatter_reducers,
                                                    create_mlp=True)

    def forward(self, graph: Batch):
        """
        Compute edge updates for the edges of the Module for homogeneous graphs in-place.
        An updated representation of the edge attributes for all edge_types is written back into the graph
        Args:
            graph: Data object of pytorch geometric. Represents a batch of homogeneous graphs
        Returns: None
        """
        source_indices, dest_indices = graph.edge_index
        edge_source_nodes = graph.x[source_indices]
        edge_dest_nodes = graph.x[dest_indices]

        aggregated_features = torch.cat([edge_source_nodes, edge_dest_nodes, graph.edge_attr], 1)

        graph.__setattr__("edge_attr", self.mlp(aggregated_features))


class HomogeneousMessagePassingNodeModule(HomogeneousMetaModule):
    """
    Module for computing node updates of a step on a homogeneous message passing GNN. Node inputs are concatenated:
    Its own Node features, the reduced features of all incoming edges.
    """

    def __init__(self, *,
                 latent_dimension: int,
                 mlp_config: Dict[str, Any],
                 scatter_reducers: List[Callable],
                 flip_edges_for_nodes: bool = False):
        """
        Module responsible for the node update of a step on a homogeneous message passing GNN.
        Node inputs are concatenated: Its own Node features, the reduced features of all incoming edges.
        Args:
            latent_dimension: Dimensionality of the internal layers of the mlp
            mlp_config: Dictionary specifying the way that the MLPs for each update should look like
            scatter_reducers: How to aggregate over the nodes/edges. Can for example be [torch.scatter_mean]
            flip_edges_for_nodes: whether to flip the edge indices for the aggregation of edge features
        """
        # use self.mlp for the update
        n_scatter_ops = len(scatter_reducers)
        in_features = latent_dimension * (1 + n_scatter_ops)  # node and aggregated incoming edge features
        create_mlp = True

        super(HomogeneousMessagePassingNodeModule, self).__init__(in_features=in_features,
                                                                  latent_dimension=latent_dimension,
                                                                  mlp_config=mlp_config,
                                                                  scatter_reducers=scatter_reducers,
                                                                  create_mlp=create_mlp)

        # use the source indices for feature aggregation if edges shall be flipped
        if flip_edges_for_nodes:
            self._get_edge_indices = lambda src_and_dest_indices: src_and_dest_indices[0]
        else:
            self._get_edge_indices = lambda src_and_dest_indices: src_and_dest_indices[1]

    def forward(self, graph: Batch):
        """
        Compute node updates for the nodes of the Module for homogeneous graphs in-place. An updated representation of
        the node attributes for all node_types is written back into the graph.
        Returns:

        """
        src_indices, dest_indices = graph.edge_index
        scatter_edge_indices = self._get_edge_indices((src_indices, dest_indices))

        aggregated_edge_features = self.multiscatter(features=graph.edge_attr, indices=scatter_edge_indices,
                                                     dim=0, dim_size=graph.x.shape[0])
        aggregated_features = torch.cat([graph.x, aggregated_edge_features], dim=1)

        # update
        graph.__setattr__("x", self.mlp(aggregated_features))


class HomogeneousGatNodeModule(torch.nn.Module):
    """
    Module for computing node updates of a step on a homogeneous message passing GNN. Node inputs are concatenated:
    Its own Node features, the reduced features of all incoming edges.
    """

    def __init__(self, *,
                 latent_dimension: int,
                 flip_edges_for_nodes: bool = False,
                 heads: int = 4,
                 ):
        """
        Module responsible for the node update of a step on a homogeneous GAT with edge updates.
        Args:
            latent_dimension: Dimensionality of the internal layers of the mlp
            flip_edges_for_nodes: whether to flip the edge indices for the aggregation of edge features
            heads: number of attention heads for the GAT
        """
        super(HomogeneousGatNodeModule, self).__init__()
        from torch_geometric.nn import GATv2Conv
        in_channels = latent_dimension
        self._gat = GATv2Conv(
            in_channels=in_channels,
            out_channels=int(latent_dimension / heads),
            heads=heads,
            add_self_loops=False,
            edge_dim=latent_dimension,
        )

        # use the source indices for feature aggregation if edges shall be flipped
        if flip_edges_for_nodes:
            self._maybe_flip_edges = lambda x: x.flip(0)
        else:
            self._maybe_flip_edges = lambda x: x

    def forward(self, graph: Batch):
        """
        Compute node updates for the nodes of the Module for homogeneous graphs in-place. An updated representation of
        the node attributes for all node_types is written back into the graph.
        Returns:

        """
        node_input = graph.x
        graph_edges = graph.edge_index
        graph_edges = self._maybe_flip_edges(graph_edges)
        graph.__setattr__("x", self._gat(node_input, graph_edges, edge_attr=graph.edge_attr))

