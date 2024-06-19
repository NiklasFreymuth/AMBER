r"""
Base class for an Abstract (Static) Finite Element Problem.
The problem specifies a partial differential equation to be solved, and the boundary conditions. It also specifies the
domain/geometry of the problem.
Currently, uses a triangular mesh with linear elements.
"""
import os
from abc import ABC

from skfem import Basis, ElementTriP1, Mesh

from src.environments.problems.abstract_fem_problem import (
    AbstractFiniteElementProblem,
)

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class AbstractFiniteElementProblem2D(AbstractFiniteElementProblem, ABC):

    def mesh_to_basis(self, mesh: Mesh) -> Basis:
        """
        Creates a basis for the given mesh.
        Args:
            mesh: The mesh to create the basis for

        Returns:
            The basis for the given mesh, including potential boundaries
        """
        return Basis(mesh, ElementTriP1())
