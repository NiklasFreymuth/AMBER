from src.algorithms.baselines.mesh_to_image import MeshImage
from src.algorithms.util.normalizers.abstract_running_normalizer import AbstractRunningNormalizer
from util.torch_util.torch_running_mean_std import TorchRunningMeanStd
from util.types import *


class MeshImageRunningNormalizer(AbstractRunningNormalizer):
    def __init__(self,
                 num_features: int,
                 normalize_features: bool,
                 observation_clip: float = 10,
                 epsilon: float = 1.0e-6,
                 device: str = "cpu",
                 ):
        """
        Normalizes the observations and predictions of an online graph-based learning algorithm
        Args:
            num_features: the number of features in the (batched) input tensor
            normalize_features: whether to normalize the features
            observation_clip: the maximum absolute value of the normalized observations
            epsilon: a small value to add to the variance to avoid division by zero

        """
        super().__init__(observation_clip=observation_clip,
                         epsilon=epsilon)

        if normalize_features:
            self.feature_normalizer = TorchRunningMeanStd(epsilon=epsilon,
                                                          shape=(num_features,),
                                                          device=device)
        else:
            self.feature_normalizer = None

    def to(self, device: str):
        if self.feature_normalizer is not None:
            self.feature_normalizer.to(device)

    def update_observation_normalizers(self, observations: MeshImage):
        """
        Update the normalizers with the given observations. Assumes that the observations are either a Data or
        HeteroData object, and that they contain a field ".y" that contains the target predictions/labels
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

    def normalize_observations(self, input_tensor: Tensor) -> InputBatch:
        """
        Normalize observations using this instances current statistics.
        Calling this method does not update statistics. It can thus be called for training as well as evaluation.
        """
        # unpack
        if self.feature_normalizer is not None:
            input_tensor = self._normalize(input_tensor=input_tensor, normalizer=self.feature_normalizer)
        return input_tensor
