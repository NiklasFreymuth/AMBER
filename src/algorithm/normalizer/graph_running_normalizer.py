from torch_geometric.data import Data

from src.algorithm.normalizer.running_normalizer import RunningNormalizer
from src.algorithm.normalizer.torch_running_mean_std import TorchRunningMeanStd
from src.algorithm.prediction_transform.prediction_transform import PredictionTransform


class GraphRunningNormalizer(RunningNormalizer):
    def __init__(
        self,
        example_graph: Data,
        normalize_inputs: bool,
        normalize_predictions: bool,
        prediction_transform: PredictionTransform,
        input_clip: float = 10,
        epsilon: float = 1.0e-6,
    ):
        """
        Normalizes the observations and predictions of an online graph-based learning algorithm.

        This class extends `RunningNormalizer` and provides normalization for graph-structured data,
        including node features, edge features, and predictions. It supports optional input normalization
        and an inverse transformation for predictions.

        Args:
            example_graph (Data):
                A PyTorch Geometric `Data` object that serves as an example input.
                It should contain the expected structure of node features (`x`), edge attributes (`edge_attr`),
                and labels (`y`).
            normalize_inputs (bool):
                Whether to normalize the input features (nodes and edges).
            normalize_predictions (bool):
                Whether to normalize the predictions (i.e., the labels `y` in the graph).
            prediction_transform (PredictionTransform):
                Has "transform" and "inverse_transform" methods to map from network space
                to mesh/sizing field space and back.
            input_clip (float, default=10):
                The maximum absolute value allowed for the normalized input features. This prevents
                extreme values from dominating the normalization.
            epsilon (float, default=1.0e-6):
                A small constant added to the variance to prevent division by zero in normalization.

        Attributes:
            -node_normalizer (TorchRunningMeanStd):
                A running mean and standard deviation tracker for node features, if `normalize_inputs` is True.
            -edge_normalizer (TorchRunningMeanStd):
                A running mean and standard deviation tracker for edge features, if `normalize_inputs` is True.

        """
        num_predictions = 1 if example_graph.y.ndim == 1 else example_graph.y.shape[1]
        super().__init__(
            num_predictions=num_predictions,
            normalize_predictions=normalize_predictions,
            prediction_transform=prediction_transform,
            input_clip=input_clip,
            epsilon=epsilon,
        )

        if normalize_inputs:
            num_node_features = example_graph.x.shape[1]
            num_edge_features = example_graph.edge_attr.shape[1]
            self.node_normalizer = TorchRunningMeanStd(epsilon=epsilon, shape=(num_node_features,))
            self.edge_normalizer = TorchRunningMeanStd(epsilon=epsilon, shape=(num_edge_features,))
        else:
            self.node_normalizer = None
            self.edge_normalizer = None

    def update_normalizers(self, inputs: Data) -> None:
        """
        Update the normalizers with the given inputs. Assumes that the inputs are a Data/graph object,
        and that they contain a field ".y" that contains the target predictions/labels.
        Args:
            inputs: A data object

        Returns:

        """
        if isinstance(inputs, Data):
            if self.node_normalizer is not None:
                # get device of the observations
                self.node_normalizer.update(inputs.x)
            if self.edge_normalizer is not None:
                self.edge_normalizer.update(inputs.edge_attr)

        # We only need to update the normalizer, and not actually normalize the predictions
        # since the predictions are not part of the observations but instead the outputs of the model
        self.update_prediction_normalizer(labels=inputs.y, baseline=inputs.current_sizing_field)

    def normalize_inputs(self, inputs: Data) -> Data:
        """
        Normalize input using this normalizer's current statistics.
        Calling this method does not update statistics. It can thus be called for training as well as evaluation.
        """
        # unpack
        if self.node_normalizer is not None:
            inputs.__setattr__(
                "x",
                self._normalize(input_tensor=inputs.x, normalizer=self.node_normalizer),
            )
        if self.edge_normalizer is not None:
            inputs.__setattr__(
                "edge_attr",
                self._normalize(
                    input_tensor=inputs.edge_attr,
                    normalizer=self.edge_normalizer,
                ),
            )
        return inputs
