import abc
import pathlib

import torch

from util.torch_util.torch_running_mean_std import TorchRunningMeanStd
from util.types import *


class AbstractRunningNormalizer(abc.ABC):
    def __init__(self,
                 observation_clip: float = 10,
                 epsilon: float = 1.0e-6,
                 ):
        """
        Abstract class for a running normalizer. Handles the normalization of predictions and defines interfaces for
        updating and normalizing observations.
        Args:
            observation_clip: the maximum absolute value of the normalized observations
            epsilon: a small value to add to the variance to avoid division by zero

        """

        self.epsilon = epsilon
        self.observation_clip = observation_clip

    @abc.abstractmethod
    def update_observation_normalizers(self, observations: InputBatch):
        raise NotImplementedError

    @abc.abstractmethod
    def normalize_observations(self, observations: InputBatch) -> InputBatch:
        raise NotImplementedError

    def _normalize(self, input_tensor: Tensor, normalizer: TorchRunningMeanStd,
                   feature_dimension: int = 1) -> Tensor:
        """
        Normalize a given input.
        Args:
            input_tensor: The input to normalize
            normalizer: The normalizer to use
            feature_dimension: The dimension along which to normalize.
                Will match the shape of the normalizer to the shape of the input tensor along this dimension.

        Returns: A normalized input

        """
        view_shape = [1]*input_tensor.ndim
        view_shape[feature_dimension] = -1
        mean = normalizer.mean.view(view_shape)
        var = normalizer.var.view(view_shape)
        scaled_observation = (input_tensor - mean) / torch.sqrt(var + self.epsilon)
        scaled_observation = torch.clip(scaled_observation, -self.observation_clip, self.observation_clip)
        return scaled_observation.float()

    def save(self, destination_path: pathlib.Path) -> None:
        """
        Saves the current normalizers to a checkpoint file.

        Args:
            destination_path: the path to checkpoint to
        Returns:

        """
        import pickle as pkl

        with destination_path.open("wb") as f:
            pkl.dump(self, f)

    def to(self, device: str):
        """
        Moves the normalizer to a new device.
        Args:
            device: The device to move the normalizer to.

        Returns:

        """
        pass

    @staticmethod
    def load(checkpoint_path: pathlib.Path) -> "AbstractRunningNormalizer":
        """
        Loads existing normalizers from a checkpoint.
        Args:
            checkpoint_path: The checkpoints directory of a previous experiment.

        Returns: A new normalizer object with the loaded normalization parameters.

        """
        import pathlib
        import pickle as pkl

        checkpoint_path = pathlib.Path(checkpoint_path)
        with checkpoint_path.open("rb") as f:  # load the file, create a new normalizer object and return it
            return pkl.load(f)
