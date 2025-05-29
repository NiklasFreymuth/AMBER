import torch

from src.algorithm.normalizer.running_normalizer import RunningNormalizer


class DummyRunningNormalizer(RunningNormalizer):
    def __init__(self):
        super().__init__(num_predictions=0, normalize_predictions=False)

    def update_normalizers(self, inputs):
        pass

    def normalize_inputs(self, inputs):
        return inputs

    def denormalize_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        return predictions
