from dataclasses import dataclass

import numpy as np

from src.tasks.domains.mesh_wrapper import MeshWrapper
from src.tasks.features.feature_provider import FeatureProvider
from src.tasks.features.fem.fem_problem import FEMProblem


@dataclass
class SourceData:
    expert_mesh: MeshWrapper  # The expert mesh that we want to learn from
    initial_mesh: MeshWrapper  # The (initial) mesh used to generate new meshes/images
    feature_provider: FeatureProvider | None = None

    @property
    def fem_problem(self) -> FEMProblem | None:
        if isinstance(self.feature_provider, FEMProblem):
            return self.feature_provider

    @property
    def boundary(self):
        return np.concatenate((self.expert_mesh.mesh.p.min(axis=1), self.expert_mesh.mesh.p.max(axis=1)), axis=0)
