import abc
from typing import Dict

import torch
from torch_geometric.data import Batch
from torch_scatter import scatter_mean

from src.mpn.common.latent_mlp import LatentMLP


class MessagePassingMetaModule(torch.nn.Module, abc.ABC):
    """
    Base class for the modules used in the GNN.
    They are used for updating node and edge features.
    """

    def __init__(self, *, in_features: int, latent_dimension: int, stack_config: Dict):
        """
        Args:
            in_features: Number of input features
            latent_dimension: Dimensionality of the internal layers of the mlp
            stack_config: Dictionary specifying the way that the gnn base should look like
        """
        super().__init__()
        mlp_config = stack_config.get("mlp")
        self._mlp = LatentMLP(in_features=in_features, latent_dimension=latent_dimension, config=mlp_config)

        self.in_features = in_features
        self.latent_dimension = latent_dimension


class MessagePassingEdgeModule(MessagePassingMetaModule):
    """
    Module for computing edge updates of a block on a message passing GNN. Edge inputs are concatenated:
    Its own edge features, and the features of the two participating nodes.
    """

    def forward(self, graph: Batch):
        """
        Compute edge updates/messages.
        An updated representation of the edge attributes for all edge_types is written back into the graph
        Args:
            graph: Data object of pytorch geometric.
        Returns: None
        """
        source_indices, dest_indices = graph.edge_index
        edge_source_nodes = graph.x[source_indices]
        edge_dest_nodes = graph.x[dest_indices]

        aggregated_features = torch.cat([edge_source_nodes, edge_dest_nodes, graph.edge_attr], 1)

        graph.__setattr__("edge_attr", self._mlp(aggregated_features))


class MessagePassingNodeModule(MessagePassingMetaModule):
    """
    Module for computing node updates/messages. Node inputs are concatenated:
    Its own Node features, and the reduced features of all incoming edges.
    """

    def forward(self, graph: Batch):
        """
        Compute updates for each node feature vector
            graph: Batch object of pytorch_geometric.data
        Returns: None. In-place operation
        """
        _, dest_indices = graph.edge_index
        aggregated_edge_features = scatter_mean(graph.edge_attr, dest_indices, dim=0, dim_size=graph.x.shape[0])
        aggregated_features = torch.cat([graph.x, aggregated_edge_features], dim=1)

        # update
        graph.__setattr__("x", self._mlp(aggregated_features))
