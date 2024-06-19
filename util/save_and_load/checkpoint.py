from dataclasses import dataclass
from typing import Optional

from src.algorithms.util.normalizers.abstract_running_normalizer import (
    AbstractRunningNormalizer,
)
from src.modules.abstract_architecture import AbstractArchitecture


@dataclass
class Checkpoint:
    architecture: AbstractArchitecture
    normalizer: Optional[AbstractRunningNormalizer]
