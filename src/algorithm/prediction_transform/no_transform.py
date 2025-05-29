import torch

from src.algorithm.prediction_transform import PredictionTransform


class NoTransform(PredictionTransform):
    def _transform(self, prediction: torch.Tensor) -> torch.Tensor:
        return prediction

    def _inverse_transform(self, prediction: torch.Tensor) -> torch.Tensor:
        return prediction
