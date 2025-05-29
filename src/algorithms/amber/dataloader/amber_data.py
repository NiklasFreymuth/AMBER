from dataclasses import dataclass
from typing import Union

from torch import Tensor
from torch_geometric.data import Data, HeteroData

from src.algorithms.amber.mesh_wrapper import MeshWrapper
from src.algorithms.baselines.mesh_to_image import MeshImage
from util.types import InputBatch


@dataclass
class MeshRefinementData:
    fem_idx: int
    mesh: MeshWrapper
    observation: Union[MeshImage, InputBatch]


@dataclass
class AMBERData(MeshRefinementData):
    refinement_depth: int = 0
    sampled_count: int = 0  # how often this piece of data has been sampled.

    @property
    def graph(self) -> Union[Data, HeteroData]:
        assert isinstance(self.observation, Data) or isinstance(self.observation, HeteroData)
        return self.observation

    # We initialize new data with the average of all data to catch up with the old data

    def increment_sampled_count(self):
        self.sampled_count += 1
