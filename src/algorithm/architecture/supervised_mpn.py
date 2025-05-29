import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch_geometric.data import Batch, Data

from src.algorithm.architecture.mlp import MLP
from src.mpn.get_message_passing_base import get_mpn_from_graph


class SupervisedMPN(nn.Module):
    def __init__(self, architecture_config: DictConfig, example_graph: Data):
        """
        Initializes the supervised MPN architecture.
        This is a simple message passing network with a supervised decoder head.

        Args:
            example_graph: Example graph to infer the input feature dimensions for nodes and edges
            architecture_config: Configuration for the policy and value networks.
        """
        super(SupervisedMPN, self).__init__()

        self._node_type = "node"
        latent_dimension = architecture_config.latent_dimension
        self.mpn = get_mpn_from_graph(
            example_graph=example_graph,
            latent_dimension=latent_dimension,
            node_name=self._node_type,
            base_config=architecture_config,
        )

        mlp_config = architecture_config.decoder
        self.decoder_mlp = MLP(
            in_features=latent_dimension,
            mlp_config=mlp_config,
            latent_dimension=latent_dimension,
        )
        self.readout = nn.Linear(latent_dimension, 1)

    def forward(self, observations: Batch, **kwargs) -> torch.Tensor:
        """

        Args:
            observations: (Batch of) observation graph(s)

        Returns:
            A scalar value for each node in the batch of graphs
        """
        node_features, _, _ = self.mpn(observations)
        node_features = node_features.get(self._node_type)

        if hasattr(observations, "mask_output"):
            node_features = node_features[observations.mask_output]
        decoded_node_features = self.decoder_mlp(node_features)
        outputs = self.readout(decoded_node_features)

        return outputs
