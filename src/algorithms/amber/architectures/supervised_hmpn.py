import torch
from src.hmpn import get_hmpn_from_graph

from src.modules.abstract_architecture import AbstractArchitecture
from src.modules.mlp import MLP
from util.types import *


class SupervisedHMPN(AbstractArchitecture):
    def __init__(
        self,
        example_graph: Union[Data, HeteroData],
        network_config: ConfigDict,
        use_gpu: bool,
    ):
        """
        Initializes the supervised HMPN architecture. This is a simple message passing network with a supervised
        decoder head.

        Args:
            example_graph: Example graph to infer the input feature dimensions for nodes, edges and globals
            network_config: Configuration for the policy and value networks.
            use_gpu: Whether to use the GPU
        """
        super(SupervisedHMPN, self).__init__(use_gpu=use_gpu, network_config=network_config)

        self._node_type = "node"
        latent_dimension = network_config.get("latent_dimension")
        base_config = network_config.get("base")
        self.hmpn = get_hmpn_from_graph(
            example_graph=example_graph,
            latent_dimension=latent_dimension,
            node_name=self._node_type,
            base_config=base_config,
            device=self._gpu_device,
        )

        mlp_config = network_config.get("decoder").get("mlp")
        self.decoder_mlp = MLP(
            in_features=latent_dimension,
            config=mlp_config,
            latent_dimension=latent_dimension,
            out_features=1,
            device=self._gpu_device,
        )

        training_config = network_config.get("training")
        self._initialize_optimizer_and_scheduler(training_config=training_config)

        self.to(self._gpu_device)

    def forward(self, observations: InputBatch, **kwargs) -> torch.Tensor:
        """

        Args:
            observations: (Batch of) observation graph(s)

        Returns:
            A scalar value for each node in the batch of graphs
        """
        observations = self.to_gpu(observations)
        node_features, _, _, batches = self.hmpn(observations)
        node_features = node_features.get(self._node_type)
        outputs = self.decoder_mlp(node_features)
        return outputs
