r"""
Abstract Base class for Poisson equations.
The poisson equation is given as \Delta u = f, where \Delta is the Laplacian, u is the solution, and f is the
load. We consider a 2D domain with zero boundary conditions.
"""
import os
from typing import Any, List, Union

import numpy as np
from omegaconf import DictConfig
from skfem import Basis, LinearForm, Mesh, asm, condense, solve
from skfem.models import laplace

from src.helpers.custom_types import PlotDict
from src.helpers.qol import wrapped_partial
from src.mesh_util.mesh_util import get_element_midpoints
from src.tasks.domains import ExtendedMeshTri1
from src.tasks.domains.mesh_wrapper import MeshWrapper
from src.tasks.features.fem.elliptic_problem.elliptic_problem import EllipticProblem
from src.tasks.features.fem.elliptic_problem.error_indicators import (
    get_edge_residual,
    get_interior_residual,
)
from src.tasks.features.fem.elliptic_problem.load_function.gmm_load import GMMDensity

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


###
# lots of complicated helper functions that are used to calculate an error indicator
###


def wrap_load(v, w, evaluate_load: callable, *args, **kwargs) -> np.ndarray:
    """
    Calculate the load for positions x and y. This is the function "f" of the rhs of the poisson equation.
    """
    x, y = w.x
    positions = np.stack((x, y), axis=-1)
    return evaluate_load(positions, *args, **kwargs) * v


class PoissonProblem(EllipticProblem):
    def __init__(
        self,
        *,
        fem_config: DictConfig[Union[str, int], Any],
        initial_mesh: ExtendedMeshTri1,
        observation_features: List[str],
        random_state: np.random.RandomState = np.random.RandomState(),
    ):
        """
        Initializes a Poisson equation with load f(x,y) parameterized by some exponential function or a
        Gaussian Mixture Model.
        Args:
            fem_config: Configuration for the finite element method. Contains
                load_function: Dictionary consisting of keys for the specific kind of load function to build
            random_state: Internally used random_state to generate domains and loads
        """
        super().__init__(
            fem_config=fem_config,
            initial_mesh=initial_mesh,
            observation_features=observation_features,
            random_state=random_state,
        )
        load_function_config = fem_config.get("load_function")
        bounding_box = np.concatenate((initial_mesh.p.min(axis=1), initial_mesh.p.max(axis=1)), axis=0)
        self._load_function = GMMDensity(
            target_function_config=load_function_config,
            bounding_box=bounding_box,
            random_state=random_state,
            valid_point_function=self._points_in_domain,
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

        return Basis(mesh, ElementTriP1())

    def _points_in_domain(self, candidate_points: np.array, distance_threshold: float = 0.0) -> np.array:
        """
        Returns a subset of points that are inside the current domain, i.e., that can be found in the mesh.
        Returns:

        """
        boundary_facets = self.initial_mesh.boundary_facets()
        boundary_node_indices = self.initial_mesh.facets[:, boundary_facets]
        line_segments = self.initial_mesh.p[:, boundary_node_indices].T.reshape(-1, 4)
        from src.tasks.features.fem.elliptic_problem.line_segment_distance import (
            get_line_segment_distances,
        )

        distances = get_line_segment_distances(candidate_points, line_segments, return_minimum=True, return_tangent_points=False)
        valid_points = candidate_points[distances > distance_threshold]
        return valid_points

    def _calculate_solution(self, basis: Basis) -> np.array:
        """
        Calculates a solution for the parameterized Poisson equation based on the given basis. The solution is
        calculated for every node/vertex of the underlying mesh, and the way it is calculated depends on the element
        used in the basis.
        For example, ElementTriP1() elements will draw 3 quadrature points for each face that lie in the middle of the
        edge between the barycenter of the face and its spanning nodes, and then linearly interpolate based on those
        elements.
        Args:

        Returns: An array (num_vertices, ), where every entry corresponds to the solution of the parameterized Poisson
            equation at the position of the respective node/vertex.

        """
        K = asm(laplace, basis)  # finite element assembly. Returns a sparse matrix
        f = asm(LinearForm(self.load), basis)  # rhs of the linear system that matches the load function

        interior = basis.mesh.interior_nodes()  # mesh nodes that are not part of the boundary

        condensed_system = condense(K, f, I=interior)  # condense system by zeroing out all nodes that lie on a boundary

        # "solve" just takes a sparse matrix and a right handside, i.e., it just solves a (linear) system of equations
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
        interior_error = get_interior_residual(basis=basis, load=self._load_function)
        edge_error = get_edge_residual(basis=basis, solution=solution[:, 0])
        error_estimate = interior_error + edge_error

        if error_estimate.ndim == 1:
            # add a dimension if the error is one-dimensional to conform to general interface
            # of (num_elements, num_solution_dimensions)
            error_estimate = error_estimate[:, None]
        return error_estimate

    # wrapper functions for the load function for the finite element assembly
    @property
    def load(self) -> callable:
        return wrapped_partial(wrap_load, evaluate_load=self.load_function)  # evaluate_load, load=self._load_function)

    @property
    def load_function(self) -> callable:
        return self._load_function.evaluate  # wrapped_partial(evaluate_load, load=self._load_function)

    ##############################
    #   Observations/Features    #
    ##############################

    def _element_problem_features(self, mesh: Mesh, observation_feature_names: List[str]) -> List[np.array]:
        """
        Returns a list of len num_features, where each entry contains num_elements features.
        Args:
            mesh: The mesh to use for the feature calculation
            observation_feature_names: The names of the features to calculate.
            Will check for these names if a corresponding feature is available.

        Returns: A list of len num_features, where each feature is of length (num_elements,)
        """
        features = []
        if "load_function" in observation_feature_names:
            features.append(self.load_function(get_element_midpoints(mesh)))
        return features

    def _problem_vertex_features(self, mesh: Mesh, observation_feature_names: List[str]) -> List[np.array]:
        """
        Returns a list of len num_features, where each entry contains num_vertices features.
        Args:
            mesh: The mesh to use for the feature calculation
            observation_feature_names: The names of the features to calculate.
                Will check for these names if a corresponding feature is available.

        Returns: A list of len num_features, where each feature is of length (num_vertices,)

        """
        features = []
        if "load_function" in observation_feature_names:
            features.append(self.load_function(mesh.p.T))
        return features

    ###############################
    # plotting utility functions #
    ###############################

    def additional_plots(self, mesh: MeshWrapper) -> PlotDict:
        """
        Build and return additional plots that are specific to this FEM problem.

        Args:
            mesh: The mesh to use for the feature calculation

        """
        load_function = np.array([self.load_function(mesh.p.T)]).T
        log_load_function = np.maximum(np.log(load_function + 1.0e-12), -10)

        from src.mesh_util.mesh_visualization import plot_mesh

        additional_plots = {
            "load_function": plot_mesh(mesh=mesh, scalars=load_function, title="Load function"),
            "log_load_function": plot_mesh(mesh=mesh, scalars=log_load_function, title="Log Load function"),
        }
        return additional_plots
