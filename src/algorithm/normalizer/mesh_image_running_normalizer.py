from typing import Callable, Optional

import torch

from src.algorithm.normalizer.running_normalizer import RunningNormalizer
from src.algorithm.normalizer.torch_running_mean_std import TorchRunningMeanStd
from src.algorithm.prediction_transform import PredictionTransform
from src.mesh_util.transforms.mesh_to_image import MeshImage


class MeshImageRunningNormalizer(RunningNormalizer):
    def __init__(
        self,
        example_mesh_image: MeshImage,
        normalize_inputs: bool,
        normalize_predictions: bool,
        prediction_transform: PredictionTransform = None,
        input_clip: float = 10,
        epsilon: float = 1.0e-6,
    ):
        """
        Normalizes the observations and predictions of an online learning algorithm for mesh-based image data.

        This class extends `RunningNormalizer` and provides normalization for mesh-structured image data,
        including feature normalization and prediction normalization. It supports optional input feature
        normalization and an inverse transformation for predictions.

        Args:
            example_mesh_image (MeshImage):
                A `MeshImage` object that represents an example input.
                It should contain `features` (input features) and `labels` (ground truth predictions).
            normalize_inputs (bool):
                Whether to normalize the input features.
            normalize_predictions (bool):
                Whether to normalize the predictions (i.e., the labels in `example_image`).
            prediction_transform (PredictionTransform):
                A function applied to the labels to handle prediction normalization and denormalization.
                If provided, this function should take a tensor as input and return a transformed tensor.
            input_clip (float, default=10):
                The maximum absolute value allowed for the normalized input features. This prevents
                extreme values from dominating the normalization.
            epsilon (float, default=1.0e-6):
                A small constant added to the variance to prevent division by zero in normalization.

        Attributes:
            -feature_normalizer (Optional[TorchRunningMeanStd]):
                A running mean and standard deviation tracker for input features if `normalize_features` is True.
                If `normalize_features` is False, this attribute is set to `None`.

        """
        num_predictions = 1 if example_mesh_image.labels.ndim == 1 else example_mesh_image.labels.shape[1]
        num_features = example_mesh_image.features.shape[1]
        super().__init__(
            num_predictions=num_predictions,
            normalize_predictions=normalize_predictions,
            prediction_transform=prediction_transform,
            input_clip=input_clip,
            epsilon=epsilon,
        )

        if normalize_inputs:
            self.feature_normalizer = TorchRunningMeanStd(
                epsilon=epsilon,
                shape=(num_features,),
            )
        else:
            self.feature_normalizer = None

    def update_normalizers(self, observations: MeshImage):
        """
        Update the normalizers with the given observations. Assumes that the observations are a Data object that
         contains a field ".y" that contains the target predictions/labels
        Args:
            observations:

        Returns:

        """
        # We only need to update the normalizer, and not actually normalize the predictions
        # since the predictions are not part of the observations but instead the outputs of the model
        if self.feature_normalizer is not None:
            features = observations.features
            assert len(features) == 1, f"Only features of batch size 1 are supported, given '{len(features)}'"
            features = features[0]
            features = features.reshape(features.shape[0], -1).T
            self.feature_normalizer.update(features)

        self.update_prediction_normalizer(labels=observations.labels)

    def normalize_inputs(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize observations using this instances current statistics.
        Calling this method does not update statistics. It can thus be called for training as well as evaluation.
        """
        # unpack
        if self.feature_normalizer is not None:
            input_tensor = self._normalize(input_tensor=input_tensor, normalizer=self.feature_normalizer)
        return input_tensor
