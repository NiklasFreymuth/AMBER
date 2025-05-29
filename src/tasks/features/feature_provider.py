r"""
Mock FEMProblem class that provides features from a file to a mesh.
"""
import abc
import os
from typing import Dict, List, Optional

import numpy as np

from src.helpers.custom_types import PlotDict
from src.tasks.domains.mesh_wrapper import MeshWrapper

if not os.name == "posix":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class FeatureProvider(abc.ABC):
    def __init__(
        self,
        *,
        observation_features: List[str],
    ):
        """
        Abstract base class for a feature provider that provides features for a mesh on either a node or element level
        Args:
            observation_features:
        """
        self._observation_feature_names = observation_features if observation_features else None

    @property
    def observation_feature_names(self) -> Optional[Dict[str, List[str]]]:
        return self._observation_feature_names

    @property
    def vertex_feature_names(self) -> List[str]:
        return self.observation_feature_names.get("vertex", [])

    @property
    def element_feature_names(self) -> List[str]:
        return self.observation_feature_names.get("element", [])

    ##############################
    #       Observations         #
    #      (Element-Level)       #
    ##############################

    def get_element_features(self, wrapped_mesh: MeshWrapper, observation_feature_names: List[str] = None) -> np.ndarray:
        """
        Returns features on the mesh elements.
        Args:
            wrapped_mesh: The mesh object to calculate the fem solution and other observations for
            observation_feature_names: The names of the node features to calculate. If None, uses the node_feature_names
                attribute of the FEM problem.

        Returns: A numpy array of shape (num_elements, num_features) containing the
            features for the mesh elements.

        """

        if observation_feature_names is None:
            observation_feature_names = self.element_feature_names

        return self._get_element_features(wrapped_mesh, observation_feature_names)

    def _get_element_features(self, wrapped_mesh: MeshWrapper, observation_feature_names: List[str]) -> np.ndarray:
        """
        Returns features on the mesh elements.
        Args:
            wrapped_mesh: The mesh object to calculate the fem solution and other observations for
            observation_feature_names: The names of the node features to calculate. If None, uses the node_feature_names
                attribute of the FEM problem.

        Returns: A numpy array of shape (num_elements, num_features) containing the
            features for the mesh elements.

        """
        raise NotImplementedError

    #############################
    #       Observations        #
    #      (Vertex-Level)       #
    #############################

    def get_vertex_features(self, wrapped_mesh: MeshWrapper, observation_feature_names: List[str] = None) -> np.ndarray:
        """
        Returns features on the mesh vertices.
        Args:
            wrapped_mesh: The mesh object to calculate the fem solution and other observations for
            observation_feature_names: The names of the node features to calculate. If None, uses the node_feature_names
                attribute of the FEM problem.

        Returns: A numpy array of shape (num_vertices, num_features) containing the features for the mesh vertices.

        """

        if observation_feature_names is None:
            observation_feature_names = self.vertex_feature_names

        return self._get_vertex_features(wrapped_mesh, observation_feature_names)

    def _get_vertex_features(self, wrapped_mesh: MeshWrapper, observation_feature_names: List[str]) -> np.ndarray:
        """
        Returns features on the mesh vertices.
        Args:
            wrapped_mesh: The mesh object to calculate the fem solution and other observations for
            observation_feature_names: The names of the node features to calculate. If None, uses the node_feature_names
                attribute of the FEM problem.

        Returns: A numpy array of shape (num_vertices, num_features) containing the features for the mesh vertices.

        """
        raise NotImplementedError

    ###############################
    # plotting utility functions #
    ###############################

    def additional_plots(self, mesh: MeshWrapper) -> PlotDict:
        """
        This function can be overwritten to add additional plots specific to the current feature provider.
        Returns:

        """
        return {}
