r"""
Abstract Base class for Poisson equations.
The poisson equation is given as \Delta u = f, where \Delta is the Laplacian, u is the solution, and f is the
load. We consider a 2D domain with zero boundary conditions.
"""
import os
from typing import Any, List, Union

import numpy as np
from omegaconf import DictConfig
from skfem import Basis, Mesh, asm, condense, solve
from skfem.models import laplace

from src.helpers.custom_types import PlotDict
from src.tasks.domains.mesh_wrapper import MeshWrapper
from src.tasks.features.fem.elliptic_problem.elliptic_problem import EllipticProblem
from src.tasks.features.fem.elliptic_problem.error_indicators import get_edge_residual
from src.tasks.features.fem.elliptic_problem.load_function.gmm_load import GMMDensity

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class LaplaceProblem(EllipticProblem):
    def __init__(
        self,
        *,
        fem_config: DictConfig[Union[str, int], Any],
        initial_mesh: Mesh,
        observation_features: List[str],
        random_state: np.random.RandomState = np.random.RandomState(),
    ):
        """
        Initializes a Poisson equation with load f(x,y) parameterized by some exponential function or a
        Gaussian Mixture Model.
        Args:
            fem_config: Configuration for the finite element method. Contains
                : Dictionary consisting of keys for the specific kind of load function to build
            random_state: Internally used random_state to generate domains and loads
        """
        super().__init__(
            fem_config=fem_config,
            initial_mesh=initial_mesh,
            observation_features=observation_features,
            random_state=random_state,
        )
        target_function_config = fem_config.get("load_function")
        self._boundary = np.concatenate((initial_mesh.p.min(axis=1), initial_mesh.p.max(axis=1)), axis=0)
        self._load_function = GMMDensity(
            target_function_config=target_function_config,
            bounding_box=self._boundary,
            random_state=random_state,
            valid_point_function=None,
            dimension=2,
        )

    def mesh_to_basis(self, mesh: Mesh) -> Basis:
        """
        Creates a basis for the given mesh.
        Args:
            mesh: The mesh to create the basis for

        Returns:
            The basis for the given mesh, including potential boundaries
        """

        from skfem import ElementTriP1

        inner_boundary_facets = self._get_inner_boundary_facets(mesh=mesh)

        mesh_ = mesh.with_boundaries(
            {
                "source": inner_boundary_facets,
            },
        )
        return Basis(mesh_, ElementTriP1())

    def _get_inner_boundary_facets(self, mesh: Mesh) -> np.array:
        """
        Returns the inner boundary facets of the mesh.
        """
        # todo: Use for features.
        inner_boundary_facets = mesh.facets_satisfying(
            lambda x: np.logical_not(
                np.logical_or.reduce(
                    (
                        np.isclose(x[0], self._boundary[0], atol=1e-10, rtol=1e-10),
                        np.isclose(x[0], self._boundary[2], atol=1e-10, rtol=1e-10),
                        np.isclose(x[1], self._boundary[1], atol=1e-10, rtol=1e-10),
                        np.isclose(x[1], self._boundary[3], atol=1e-10, rtol=1e-10),
                    )
                )
            ),
            boundaries_only=True,
        )
        return inner_boundary_facets

    def _calculate_solution(self, basis: Basis) -> np.array:
        """ """
        boundary_temperature = basis.zeros()
        boundary_dofs = basis.get_dofs({"source"})
        boundary_positions = boundary_dofs.doflocs[:, boundary_dofs.nodal_ix].T
        boundary_temperature[boundary_dofs] = self._load_function.evaluate(samples=boundary_positions)

        # Assemble matrices, solve problem
        matrix = asm(laplace, basis)
        condensed_system = condense(matrix, x=boundary_temperature, I=basis.mesh.interior_nodes())
        solution = solve(*condensed_system)
        return solution

    def get_error_indicator(self, mesh: Mesh, solution: np.array) -> np.array:
        """
        Calculates the error indicator for the given basis and solution.
        Args:
            mesh: The mesh to use for the error calculation
            solution: The solution to use for the error calculation

        Returns: An array of shape (num_elements, num_solution_dimensions) containing the error indicator for each element

        """
        basis = self.mesh_to_basis(mesh)
        error_estimate = get_edge_residual(basis=basis, solution=solution[:, 0])
        if error_estimate.ndim == 1:
            # add a dimension if the error is one-dimensional to conform to general interface
            # of (num_elements, num_solution_dimensions)
            error_estimate = error_estimate[:, None]
        return error_estimate

    ##############################
    #   Observations/Features    #
    ##############################

    def _element_problem_features(self, mesh: Mesh, observation_feature_names: List[str]) -> List[np.array]:
        """
        Returns a list of len num_features, where each entry contains num_elements features.
        """
        return []

    def _problem_vertex_features(self, mesh: Mesh, observation_feature_names: List[str]) -> List[np.array]:
        """
        Returns a list of len num_features, where each entry contains num_vertices features.
        """
        return []

    ###############################
    # plotting utility functions #
    ###############################

    def additional_plots(self, mesh: MeshWrapper) -> PlotDict:
        return {}
