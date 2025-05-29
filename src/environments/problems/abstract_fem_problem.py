r"""
Wrapper of a given finite element problem.
In particular, the FEM Problem consists of an original coarse mesh and basis, and a fine-grained mesh, basis,
and solution.
"""
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import plotly.graph_objects as go
from skfem import Basis, Mesh

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class AbstractFiniteElementProblem:
    def __init__(
            self,
            *,
            fem_config: Dict[Union[str, int], Any],
            initial_mesh: Mesh,
            element_features: List[str],
            random_state: np.random.RandomState = np.random.RandomState(),
    ):
        """
        This class stores all the information about the problem itself, as well as temporary information about the
        current mesh, and solution.
        It also provides interfaces to, e.g., plot details about the problem.
        Args:
            fem_config: Configuration of the finite element problem
            initial_mesh: The initial mesh to use for the problem
            element_features: Features to extract from the PDE
            random_state: The random state to use for the problem
        """
        self._random_state = random_state
        self._fem_config = fem_config

        #################
        # problem state #
        #################
        self._pde_element_feature_names = element_features if element_features else None

        ###################
        # mesh parameters #
        ###################
        self.initial_mesh: Mesh = initial_mesh

        #####################
        # plotting utility #
        ####################
        if self.initial_mesh.dim() == 2:
            self._plot_boundary = np.array(fem_config.get("domain").get("boundary", [0, 0, 1, 1]))
        else:
            self._plot_boundary = None

        self._set_pde()  # set pde after domain, since the domain may be used to generate the pde.

    def _set_pde(self) -> None:
        """
        Initializes a random PDE. This is not a reset, but a random initialization of the PDE.
        Returns:

        """
        raise NotImplementedError("AbstractFiniteElementProblem does not implement _set_pde()")

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

    def calculate_solution(self, mesh) -> np.ndarray:
        """
        Calculates a solution of the underlying PDE for the given finite element basis, and caches the solution
        for plotting.
        Args:

        """
        from src.algorithms.amber.mesh_wrapper import MeshWrapper
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

    @property
    def element_feature_names(self) -> Optional[List[str]]:
        return self._pde_element_feature_names

    ##############################
    #         Observations       #
    ##############################

    def element_features(self, mesh) -> np.ndarray:
        """
        Returns a dictionary of element features that are used as observations for the  RL agents.
        Args:

        Returns: An array (num_elements, num_features) that contains the features for each element of the mesh

        """
        return self._element_features(mesh=mesh, element_feature_names=self._pde_element_feature_names)

    def _element_features(self, mesh: Mesh, element_feature_names: List[str]) -> Optional[np.array]:
        """
        Returns an array of shape (num_elements, num_features) containing the features for each element.
        Args:
            mesh: The mesh to use for the feature calculation
            element_feature_names: The names of the features to calculate. Will check for these names if a corresponding
            feature is available.

        Returns: An array of shape (num_elements, num_features) containing the features for each element.

        """
        raise NotImplementedError

    def project_to_scalar(self, values: np.array) -> np.array:
        """
        Projects a value per node and solution dimension to a scalar value per node.
        Usually, takes the norm of the solution vector.
        Args:
            values: A vector of shape (num_vertices, solution_dimension)

        Returns: A scalar value per vertex
        """
        return np.linalg.norm(values, axis=1)

    ###############################
    # plotting utility functions #
    ###############################

    def additional_plots(self, mesh: Mesh) -> Dict[str, go.Figure]:
        """
        This function can be overwritten to add additional plots specific to the current FEM problem.
        Returns:

        """
        return {}

    @property
    def plot_boundary(self) -> Optional[np.ndarray]:
        return self._plot_boundary
