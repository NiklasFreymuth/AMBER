from src.algorithms.util.normalizers.abstract_running_normalizer import AbstractRunningNormalizer
from util.torch_util.torch_running_mean_std import TorchRunningMeanStd
from util.types import *


class GraphRunningNormalizer(AbstractRunningNormalizer):
    def __init__(self,
                 num_node_features: Union[int, Dict[str, int]],
                 num_edge_features: Union[int, Dict[str, int]],
                 num_global_features: Optional[int],
                 normalize_nodes: bool,
                 normalize_edges: bool,
                 normalize_globals: bool,
                 observation_clip: float = 10,
                 epsilon: float = 1.0e-6,
                 device: str = "cpu",
                 ):
        """
        Normalizes the observations and predictions of an online graph-based learning algorithm
        Args:
            num_node_features: the number of node features or a dictionary of the number of node features per node type
            num_edge_features: the number of edge features or a dictionary of the number of edge features per edge type
            num_global_features: the number of global features
            normalize_nodes: whether to normalize the node features
            normalize_edges: whether to normalize the edge features
            normalize_globals: whether to normalize the global features
            observation_clip: the maximum absolute value of the normalized observations
            epsilon: a small value to add to the variance to avoid division by zero

        """
        super().__init__(observation_clip=observation_clip,
                         epsilon=epsilon)

        if normalize_nodes:
            if isinstance(num_node_features, int):
                self.node_normalizers = TorchRunningMeanStd(epsilon=epsilon, shape=(num_node_features,), device=device)
            elif isinstance(num_node_features, dict):
                self.node_normalizers = {}
                for node_type, num_node_features in sorted(num_node_features.items()):
                    self.node_normalizers[node_type] = TorchRunningMeanStd(
                        epsilon=epsilon, shape=(num_node_features,), device=device
                    )
            else:
                raise ValueError(f"Unknown type for num_node_features: {type(num_node_features)}")
        else:
            self.node_normalizers = None

        if normalize_edges:
            if isinstance(num_edge_features, int):
                self.edge_normalizers = TorchRunningMeanStd(epsilon=epsilon, shape=(num_edge_features,), device=device)
            elif isinstance(num_edge_features, dict):
                self.edge_normalizers = {}
                for edge_type, num_edge_features in sorted(num_edge_features.items()):
                    self.edge_normalizers[edge_type] = TorchRunningMeanStd(
                        epsilon=epsilon, shape=(num_edge_features,), device=device
                    )
            else:
                raise ValueError(f"Unknown type for num_edge_features: {type(num_edge_features)}")
        else:
            self.edge_normalizers = None

        if normalize_globals:
            self.global_normalizer = TorchRunningMeanStd(epsilon=epsilon, shape=(num_global_features,), device=device)
        else:
            self.global_normalizer = None

    def to(self, device: str):
        if self.node_normalizers is not None:
            if isinstance(self.node_normalizers, dict):
                for node_normalizer in self.node_normalizers.values():
                    node_normalizer.to(device)
            else:
                self.node_normalizers.to(device)
        if self.edge_normalizers is not None:
            if isinstance(self.edge_normalizers, dict):
                for edge_normalizer in self.edge_normalizers.values():
                    edge_normalizer.to(device)
            else:
                self.edge_normalizers.to(device)
        if self.global_normalizer is not None:
            self.global_normalizer.to(device)

    def update_observation_normalizers(self, observations: InputBatch):
        """
        Update the normalizers with the given observations. Assumes that the observations are either a Data or
        HeteroData object, and that they contain a field ".y" that contains the target predictions/labels
        Args:
            observations:

        Returns:

        """
        if isinstance(observations, Data):
            if self.node_normalizers is not None:
                # get device of the observations
                self.node_normalizers.update(observations.x)
            if self.edge_normalizers is not None:
                self.edge_normalizers.update(observations.edge_attr)

        elif isinstance(observations, HeteroData):
            if self.node_normalizers is not None:
                for node_type, node_features in zip(observations.node_types, observations.node_stores):
                    self.node_normalizers[node_type].update(node_features.x)
            if self.edge_normalizers is not None:
                for edge_type, edge_features in zip(observations.edge_types, observations.edge_stores):
                    self.edge_normalizers[edge_type].update(edge_features.edge_attr)

        if self.global_normalizer is not None:
            self.global_normalizer.update(observations.u)

    def normalize_observations(self, observations: InputBatch) -> InputBatch:
        """
        Normalize observations using this instances current statistics.
        Calling this method does not update statistics. It can thus be called for training as well as evaluation.
        """
        # unpack
        if isinstance(observations, Data):
            if self.node_normalizers is not None:
                observations.__setattr__(
                    "x",
                    self._normalize(input_tensor=observations.x, normalizer=self.node_normalizers),
                )
            if self.edge_normalizers is not None:
                observations.__setattr__(
                    "edge_attr",
                    self._normalize(
                        input_tensor=observations.edge_attr,
                        normalizer=self.edge_normalizers,
                    ),
                )

        elif isinstance(observations, HeteroData):
            if self.node_normalizers is not None:
                for position, node_type in enumerate(observations.node_types):
                    observations.node_stores[position]["x"] = self._normalize(
                        input_tensor=observations.node_stores[position]["x"],
                        normalizer=self.node_normalizers[node_type],
                    )
            if self.edge_normalizers is not None:
                for position, edge_type in enumerate(observations.edge_types):
                    observations.edge_stores[position]["edge_attr"] = self._normalize(
                        input_tensor=observations.edge_stores[position]["edge_attr"],
                        normalizer=self.edge_normalizers[edge_type],
                    )

        if self.global_normalizer is not None:
            observations.__setattr__(
                "u",
                self._normalize(input_tensor=observations.u, normalizer=self.global_normalizer),
            )

        return observations
