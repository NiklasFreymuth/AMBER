from src.algorithms.util.normalizers.abstract_running_normalizer import (
    AbstractRunningNormalizer,
)
from util.types import *


class DummyRunningNormalizer(AbstractRunningNormalizer):
    def __init__(self):
        super().__init__()

    def update_observation_normalizers(self, observations: InputBatch) -> None:
        pass

    def normalize_observations(self, observations: InputBatch) -> InputBatch:
        return observations
