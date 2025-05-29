from dataclasses import dataclass
from functools import cached_property
from typing import List, Optional, get_args

import numpy as np
from torch_geometric.data import Data

from src.algorithm.dataloader.source_data import SourceData
from src.helpers.custom_types import MeshNodeType, SizingFieldInterpolationType
from src.mesh_util.transforms.mesh_to_image import MeshImage
from src.tasks.domains.mesh_wrapper import MeshWrapper
from src.tasks.features.feature_provider import FeatureProvider
from src.tasks.features.fem.fem_problem import FEMProblem


@dataclass
class MeshGenerationData:
    # A mesh generation algorithm Data point. This point contains
    # * the original "source" data, which is a mesh, a geometry, and potentially a FEM Problem
    # * the current mesh, as well as labels on this mesh and an observation derived from it
    # * config parameters
    mesh: MeshWrapper
    source_data: SourceData  # The data this data was derived from

    node_feature_names: List[str]
    node_type: MeshNodeType
    sizing_field_interpolation_type: SizingFieldInterpolationType

    def __post_init__(self):
        assert self.sizing_field_interpolation_type in get_args(
            SizingFieldInterpolationType
        ), f"{self.sizing_field_interpolation_type=} not recognized"
        self._observation: Optional[Data | MeshImage] = None

    @property
    def observation(self) -> Data | MeshImage:
        raise NotImplementedError

    @cached_property
    def _labels(self) -> np.ndarray:
        """
        Get the labels for the current mesh.
        This is the difference between the learned mesh and the expert mesh, as defined by the interpolation type

        Returns:
            np.ndarray: The labels for the mesh
        """
        from src.algorithm.util.interpolate_sizing_field import interpolate_sizing_field

        labels = interpolate_sizing_field(
            queried_mesh=self.mesh,
            fine_mesh=self.expert_mesh,
            sizing_field_interpolation_type=self.sizing_field_interpolation_type,
        )
        return labels

    @property
    def expert_mesh(self) -> MeshWrapper:
        return self.source_data.expert_mesh

    @property
    def fem_problem(self) -> FEMProblem | None:
        return self.source_data.fem_problem

    @property
    def feature_provider(self) -> FeatureProvider | None:
        return self.source_data.feature_provider
