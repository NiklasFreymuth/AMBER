r"""
Wrapper of a given finite element problem.
In particular, the FEM Problem consists of an original coarse mesh and basis, and a fine-grained mesh, basis,
and solution.
"""
import abc
import os
from typing import Any, List, Optional, Union

import numpy as np
from omegaconf import DictConfig
from skfem import Basis, Mesh

from src.helpers.custom_types import MetricDict
from src.helpers.qol import safe_concatenate
from src.tasks.domains.mesh_wrapper import MeshWrapper
from src.tasks.features.feature_provider import FeatureProvider

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class FEMProblem(FeatureProvider, abc.ABC):
    def __init__(
        self,
        *,
        fem_config: DictConfig[Union[str, int], Any],
        initial_mesh: Mesh,
        observation_features: List[str],
        random_state: np.random.RandomState = np.random.RandomState(),
    ):
        """
        This class stores all the information about the problem itself, as well as temporary information about the
        current mesh, and solution.
        It also provides interfaces to, e.g., plot details about the problem.
        Args:
            fem_config: Configuration of the finite element problem
            initial_mesh: The initial mesh to use for the problem
            observation_features: Features to extract from the PDE
            random_state: The random state to use for the problem
        """
        self._random_state = random_state
        self._fem_config = fem_config

        #################
        # problem state #
        #################
        self.initial_mesh: Mesh = initial_mesh
        super().__init__(observation_features=observation_features)

    def mesh_to_basis(self, mesh: Mesh) -> Basis:
        """
        Creates a basis for the given mesh.
        By default, uses a linear triangular basis and no boundary on the mesh.
        Args:
            mesh: The mesh to create the basis for

        Returns:
            The basis for the given mesh, including potential boundaries
        """
        raise NotImplementedError("AbstractFiniteElementProblem does not implement mesh_to_basis()")

    def calculate_solution(self, mesh: Mesh | MeshWrapper) -> np.ndarray:
        """
        Calculates a solution of the underlying PDE for the given finite element basis, and caches the solution
        for plotting.
        Args:

        """
        from src.tasks.domains.mesh_wrapper import MeshWrapper

        if isinstance(mesh, MeshWrapper):
            mesh = mesh.mesh
        solution = self._calculate_solution(basis=self.mesh_to_basis(mesh))
        if solution.ndim == 1:  # add a dimension if the solution is one-dimensional for consistency with interface
            solution = solution[:, None]
        return solution

    def _calculate_solution(self, basis: Basis) -> np.ndarray:
        """
        Calculates a solution of the underlying PDE for the given finite element basis.
        The solution is calculated for every *node/vertex* of the underlying mesh. The way it is calculated depends on
        the element used in the basis. Here, we use, ElementTriP1() elements, which are linear elements that will draw
        3 quadrature points for each element. These points lie in the middle of the edge between the barycenter of the
        element and its spanning nodes.
        The evaluation of the element is linearly interpolated based on those elements.

        Args:
            basis: The basis to calculate the solution for

        Returns: An array (num_vertices, ), where every entry corresponds to the solution of the parameterized Poisson
            equation at the position of the respective node/vertex.


        """
        raise NotImplementedError("AbstractFiniteElementProblem does not implement _calculate_solution")

    ##############################
    #       Observations         #
    #      (Element-Level)       #
    ##############################

    def _get_element_features(self, wrapped_mesh: MeshWrapper, observation_feature_names: List[str]) -> np.ndarray:
        """
        Returns fem-specific observations on the mesh elements. These include
        * the solution of the PDE. For "elements", the solution is the solution mean and std of the spanning vertices of
            the element.
            Solutions can be scalar- or vector-valued.
        * problem-specific features for the vertices or elements of the mesh. This can be boundary or process conditions,
            or other features that are relevant for the problem.
        Args:
            wrapped_mesh: The mesh object to calculate the fem solution and other observations for
            observation_feature_names: The names of the node features to calculate.

        Returns: A numpy array of shape (num_elements, num_features) containing the
            observations for the mesh.

        """

        # Add solution of the FEM problem.
        element_solution_features = self.element_solution_features(wrapped_mesh, observation_feature_names)

        # Add problem-specific FEM features to it
        element_problem_features = self.element_problem_features(mesh=wrapped_mesh.mesh, observation_feature_names=observation_feature_names)
        # Use save_concatenate to handle both empty and non-empty cases
        element_features = safe_concatenate([element_solution_features, element_problem_features], axis=1)
        return element_features

    def element_solution_features(self, wrapped_mesh: MeshWrapper, observation_feature_names: List[str]) -> Optional[np.ndarray]:
        """
        Returns the solution of the FEM problem on the elements of the wrapped mesh.
        The solution is calculated as the mean and standard deviation of the solution at the vertices of the element.
        The return is an array of shape (num_elements, num_features), where num_features is the number of solutions
        * {mean/std}
        Args:
            wrapped_mesh:
            observation_feature_names:

        Returns:

        """
        from src.mesh_util.mesh_util import get_aggregation_per_element

        solution_features = []
        # Get solution features.
        # If we use solution features, we have to solve the underlying PDE problem for the current mesh discretization
        if any("solution" in feature_name for feature_name in observation_feature_names):
            solution = self.calculate_solution(mesh=wrapped_mesh)
            if any("solution_mean" in node_feature_name for node_feature_name in observation_feature_names):
                # has at least 1 solution dimension
                assert (
                    len([x for x in observation_feature_names if "solution_mean" in x]) == solution.shape[1]
                ), "Number of solution mean features must match number of solution dimensions"
                solution_mean = get_aggregation_per_element(
                    solution=solution,
                    element_indices=wrapped_mesh.element_indices,
                    aggregation_function_str="mean",
                )
                for solution_dimension in range(solution_mean.shape[1]):
                    solution_features.append(solution_mean[:, solution_dimension])
            if any("solution_std" in node_feature_name for node_feature_name in observation_feature_names):
                assert (
                    len([x for x in observation_feature_names if "solution_std" in x]) == solution.shape[1]
                ), "Number of solution std features must match number of solution dimensions"
                solution_std = get_aggregation_per_element(
                    solution=solution,
                    element_indices=wrapped_mesh.element_indices,
                    aggregation_function_str="std",
                )
                for solution_dimension in range(solution_std.shape[1]):
                    solution_features.append(solution_std[:, solution_dimension])
        # Convert solution features to a NumPy array if not empty, else set it as None
        solution_features = np.array(solution_features).T if solution_features else None
        return solution_features

    def element_problem_features(self, mesh: Mesh, observation_feature_names: List[str] = None) -> Optional[np.array]:
        """
        Returns a dictionary of problem-specific element features that are used as part of the observation graph.
        Args:

        Returns: An array (num_elements, num_features) that contains the features for each element of the mesh

        """
        if observation_feature_names is None:
            observation_feature_names = self.element_feature_names

        features = self._element_problem_features(mesh, observation_feature_names)

        if len(features) == 0:
            return None
        else:
            return np.array(features).T

    def _element_problem_features(self, mesh: Mesh, observation_feature_names: List[str]) -> List[np.array]:
        raise NotImplementedError

    #############################
    #       Observations        #
    #      (Vertex-Level)       #
    #############################

    def _get_vertex_features(self, wrapped_mesh: MeshWrapper, observation_feature_names: List[str]) -> np.ndarray:
        """
        Returns fem-specific observations on the mesh vertices. These include
        * the solution of the PDE. For "vertices", the solution is the solution at the vertices of the mesh.
            Solutions can be scalar- or vector-valued.
        * problem-specific features for the vertices or elements of the mesh. This can be boundary or process conditions,
            or other features that are relevant for the problem.
        Args:
            wrapped_mesh: The mesh object to calculate the fem solution and other observations for
            observation_feature_names: The names of the node features to calculate. If None, uses the node_feature_names
                attribute of the FEM problem.

        Returns: A numpy array of shape (num_vertices, num_features) containing the
            observations for the mesh vertices.

        """
        vertex_solution_features = self.vertex_solution_features(wrapped_mesh, observation_feature_names)

        # Add problem-specific FEM features to it
        problem_vertex_features = self.problem_vertex_features(mesh=wrapped_mesh.mesh, observation_feature_names=observation_feature_names)
        # Use save_concatenate to handle both empty and non-empty cases
        vertex_features = safe_concatenate([vertex_solution_features, problem_vertex_features], axis=1)
        return vertex_features

    def vertex_solution_features(self, wrapped_mesh: MeshWrapper, observation_feature_names: List[str]) -> Optional[np.ndarray]:
        vertex_solution_features = []
        # Get solution features.
        # If we use solution features, we have to solve the underlying PDE problem for the current mesh discretization
        if "solution" in observation_feature_names:
            solution = self.calculate_solution(mesh=wrapped_mesh)
            for solution_dimension in range(solution.shape[1]):
                vertex_solution_features.append(solution[:, solution_dimension])
        # Convert vertex_features to a NumPy array if not empty, else set it as None
        vertex_solution_features = np.array(vertex_solution_features).T if vertex_solution_features else None
        return vertex_solution_features

    def problem_vertex_features(self, mesh: Mesh, observation_feature_names: List[str] = None) -> Optional[np.array]:
        """
        Returns a dictionary of element features that are used as part of the observation graph.
        Args:

        Returns: An array (num_vertices, num_features) that contains the features for each element of the mesh

        """
        if observation_feature_names is None:
            observation_feature_names = self.vertex_feature_names

        features = self._problem_vertex_features(mesh, observation_feature_names)
        if len(features) == 0:
            return None
        else:
            return np.array(features).T

    def _problem_vertex_features(self, mesh: Mesh, observation_feature_names: List[str]) -> List[np.array]:
        raise NotImplementedError

    ########################
    # metrics & evaluation #
    ########################

    def get_quality_metrics(self, mesh: MeshWrapper) -> MetricDict:
        raise NotImplementedError("get_quality_metrics is not implemented")
