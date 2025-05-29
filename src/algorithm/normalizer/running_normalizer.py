import abc
from typing import Optional

import torch
from torch import nn

from src.algorithm.normalizer.torch_running_mean_std import TorchRunningMeanStd
from src.algorithm.prediction_transform.prediction_transform import PredictionTransform


class RunningNormalizer(nn.Module, abc.ABC):
    def __init__(
        self,
        num_predictions: int,
        normalize_predictions: bool,
        prediction_transform: Optional[PredictionTransform] = None,
        input_clip: float = 10,
        epsilon: float = 1.0e-6,
    ):
        """
        Abstract class for a running normalizer. Handles the normalization of predictions and defines interfaces for
        updating and normalizing observations.
        Args:
            num_predictions: the number of predictions, i.e., dimensionality of the labels
            normalize_predictions: whether to normalize the predictions
            prediction_transform (PredictionTransform):
                Has "transform" and "inverse_transform" methods to map from network space
                to mesh/sizing field space and back.
            input_clip: the maximum absolute value of the normalized observations
            epsilon: a small value to add to the variance to avoid division by zero

        """
        super().__init__()

        if normalize_predictions:
            self.prediction_normalizer = TorchRunningMeanStd(epsilon=epsilon, shape=(num_predictions,))
        else:
            self.prediction_normalizer = None

        if prediction_transform is not None:
            self.prediction_transform = prediction_transform
        else:
            self.prediction_transform = None
        self.epsilon = epsilon
        self.input_clip = input_clip

    @abc.abstractmethod
    def update_normalizers(self, inputs):
        """
        Update the normalizers with the given inputs. Assumes that the inputs also contain a label, which can
        be used to update the prediction normalizer.
        Args:
            inputs: A data object

        Returns:

        """
        raise NotImplementedError

    @abc.abstractmethod
    def normalize_inputs(self, inputs):
        raise NotImplementedError

    def update_prediction_normalizer(self, labels: torch.Tensor, baseline: torch.Tensor = None) -> None:
        """
        Update the normalizers with the given labels to enable learning in a label-transformed space.
        We only need to update the normalizer, and not actually normalize the predictions since the predictions
        are not part of the observations but instead the outputs of the model.
        Applies the inverse prediction transform if it exists.
        Args:
            labels: The labels to update the normalizer with
            baseline: The baseline tensor to add to the prediction iff prediction_normalizer.predict_residual is True.

        Returns:

        """
        if self.prediction_normalizer is not None:
            if self.prediction_transform is not None:
                labels = self.prediction_transform.inverse(labels, baseline=baseline)
            self.prediction_normalizer.update(labels)

    def denormalize_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        if self.prediction_normalizer is not None:
            predictions = self._denormalize(input_tensor=predictions, normalizer=self.prediction_normalizer)
        return predictions

    def _normalize(self, input_tensor: torch.Tensor, normalizer: TorchRunningMeanStd, feature_dimension: int = 1) -> torch.Tensor:
        """
        Normalize a given input.
        This is the inverse of the self._denormalize method, except that it clips the values to the range
        Args:
            input_tensor: The input to normalize
            normalizer: The normalizer to use
            feature_dimension: The dimension along which to normalize.
                Will match the shape of the normalizer to the shape of the input tensor along this dimension.

        Returns: A normalized input

        """
        view_shape = [1] * input_tensor.ndim
        view_shape[feature_dimension] = -1
        mean = normalizer.mean.view(view_shape)
        var = normalizer.var.view(view_shape)
        scaled_observation = (input_tensor - mean) / torch.sqrt(var + self.epsilon)
        scaled_observation = torch.clip(scaled_observation, -self.input_clip, self.input_clip)
        return scaled_observation.float()

    def _denormalize(self, input_tensor: torch.Tensor, normalizer: TorchRunningMeanStd) -> torch.Tensor:
        """
        Denormalize a given input.
        This is the inverse of the self._normalize method.
        Args:
            input_tensor: The input to denormalize
            normalizer: The normalizer to use

        Returns: A denormalized input

        """
        return (input_tensor * torch.sqrt(normalizer.var + self.epsilon) + normalizer.mean).type(input_tensor.dtype)
