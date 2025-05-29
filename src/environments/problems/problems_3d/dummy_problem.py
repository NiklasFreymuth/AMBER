from typing import List, Optional

import numpy as np
from skfem import Basis, Mesh

from src.environments.problems.problems_3d.abstract_finite_element_problem_3d import (
    AbstractFiniteElementProblem3D,
)


class DummyProblem(AbstractFiniteElementProblem3D):

    def _set_pde(self) -> None:
        pass

    def _calculate_solution(self, basis: Basis) -> np.array:
        return np.zeros(basis.mesh.nvertices)

    def _element_features(self, mesh: Mesh, element_feature_names: List[str]) -> Optional[np.array]:
        return None
