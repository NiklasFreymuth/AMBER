import torch

from src.algorithm.prediction_transform import PredictionTransform


class SoftplusTransform(PredictionTransform):
    def _transform(self, prediction: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(prediction)

    def _inverse_transform(self, prediction: torch.Tensor) -> torch.Tensor:
        return prediction + torch.log(-torch.expm1(-prediction))
