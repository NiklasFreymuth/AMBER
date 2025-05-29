r"""
Class that loads a given .txt file with an inlet position and provides according features to the mesh.
"""
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.tasks.domains.mesh_wrapper import MeshWrapper
from src.tasks.features.feature_provider import FeatureProvider


class InletFeatureProvider(FeatureProvider):
    def __init__(
        self,
        *,
        inlet_file: str | Path,
        observation_features: List[str],
    ):
        inlet_file = Path(inlet_file)
        with inlet_file.open("r") as f:
            try:
                self.inlet_position = np.array([float(line.strip()) for line in f if line.strip()])
            except ValueError:
                warnings.warn("Could not parse inlet position from file. Setting to None.")
                self.inlet_position = None
        super().__init__(observation_features=observation_features)

    ##############################
    #       Observations         #
    #      (Element-Level)       #
    ##############################

    def _get_element_features(self, wrapped_mesh: MeshWrapper, observation_feature_names: List[str]) -> Optional[np.ndarray]:
        """
        Computes features from loaded element positions for the current mesh. Returns None if no features are requested, or
        an array of shape (num_elements, num_features) if features are requested.
        """
        if observation_feature_names is None:
            observation_feature_names = self.element_feature_names

        features = []
        if self.inlet_position is not None:
            if "inlet_element" in observation_feature_names:
                element_idx = wrapped_mesh.find_closest_elements(np.array([self.inlet_position]))[0]
                inlet_elements = np.zeros(wrapped_mesh.num_elements, dtype=np.float32)
                inlet_elements[element_idx] = 1.0
                features.append(inlet_elements)
        if len(features) == 0:
            return None
        else:
            return np.vstack(features).T  # final shape (num_elements, num_features)

    #############################
    #       Observations        #
    #      (Vertex-Level)       #
    #############################

    def _get_vertex_features(self, wrapped_mesh: MeshWrapper, observation_feature_names: List[str]) -> Optional[np.ndarray]:
        """
        Computes features for each vertex in the mesh. Returns None if no features are requested,
        otherwise returns an array of shape (num_vertices, num_features).
        """
        if observation_feature_names is None:
            observation_feature_names = self.vertex_feature_names

        features = []
        if self.inlet_position is not None:
            if "inlet_vertex" in observation_feature_names:
                vertex_idx = wrapped_mesh.vertex_tree.query(np.array([self.inlet_position]), k=1)[1][0]
                inlet_vertices = np.zeros(wrapped_mesh.num_vertices, dtype=np.float32)
                inlet_vertices[vertex_idx] = 1.0
                features.append(inlet_vertices)

        if not features:
            return None
        else:
            return np.vstack(features).T  # shape (num_vertices, num_features)
