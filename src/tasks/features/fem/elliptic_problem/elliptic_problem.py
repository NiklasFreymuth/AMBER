r"""
Abstract Base class for Elliptic Problems.
"""
import abc
import os

import numpy as np
from skfem import Mesh

from src.helpers.custom_types import MetricDict
from src.tasks.domains.mesh_wrapper import MeshWrapper
from src.tasks.features.fem.fem_problem import FEMProblem

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class EllipticProblem(FEMProblem, abc.ABC):
    def get_error_indicator(self, mesh: Mesh, solution: np.array) -> np.array:
        raise NotImplementedError

    ########################
    # metrics & evaluation #
    ########################

    def get_quality_metrics(self, mesh: MeshWrapper) -> MetricDict:
        solution = self.calculate_solution(mesh=mesh)
        error_indicator = self.get_error_indicator(mesh=mesh.mesh, solution=solution)
        error_indicator_sum = np.sum(error_indicator)
        error_indicator_norm = np.sqrt(error_indicator_sum)
        return {
            "error_indicator_sum": error_indicator_sum,
            "error_indicator_per_element": error_indicator_sum * mesh.num_elements,
            "error_indicator_norm": error_indicator_norm,
        }
