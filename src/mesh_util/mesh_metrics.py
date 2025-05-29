from functools import cached_property
from typing import Dict, Optional, Tuple

import numpy as np
from omegaconf import DictConfig

from src.helpers.custom_types import MetricDict
from src.helpers.qol import prefix_keys
from src.mesh_util.sizing_field_util import get_sizing_field
from src.tasks.domains.mesh_wrapper import MeshWrapper
from src.tasks.features.fem.fem_problem import FEMProblem


class MeshMetrics:
    """
    Utility class that computes similarity metrics between two meshes. Contains different functions that define
    (relative) quality and similarity metrics between a reference mesh and an evaluated mesh.
    """

    def __init__(self, metric_config: DictConfig, reference_mesh: MeshWrapper, evaluated_mesh: MeshWrapper, fem_problem: Optional[FEMProblem]):
        self.metric_config = metric_config
        self.reference_mesh = reference_mesh
        self.evaluated_mesh = evaluated_mesh
        self.fem_problem = fem_problem

    def __call__(self) -> MetricDict:
        metrics = self.get_similarity_metrics()

        if self.fem_problem is not None:
            metrics |= self.get_fem_metrics()
        return metrics

    def get_similarity_metrics(self) -> MetricDict:
        """
        Compute similarity metrics between two meshes.

        The following metrics are computed on element midpoints and/or mesh vertices:
        - The (symmetric) Chamfer distance between the evaluated mesh and the reference mesh, as well as its
        exponentiated and density-aware variants
        - The (symmetric) Earth Mover's Distance (EMD) between the evaluated mesh and the reference mesh-

        Additionally, the following metrics are computed:
        - The element size difference between the evaluated mesh and the reference mesh. These differences are
            calculated as the absolute, squared, mean and maximum differences in element size.
            We calculate them both in the original and in the log space, and either for the original mesh only or
            symmetrized between the two meshes.

        Finally, we take the difference in the number of elements between the two meshes.

        Returns: A dictionary of similarity metrics as key-value pairs

        """
        computed_metrics = {}

        pointcloud_metrics = self.metric_config.pointcloud
        pointcloud_distances = self.pointcloud_distances(pointcloud_metrics)
        computed_metrics = computed_metrics | pointcloud_distances

        if self.metric_config.projected_l2_error:
            computed_metrics["projected_l2_error"] = self.projected_l2_error()
            computed_metrics["projected_l2_error_reverse"] = self.projected_l2_error(reverse=True)
            computed_metrics["projected_l2_error_symmetric"] = (
                computed_metrics["projected_l2_error"] + computed_metrics["projected_l2_error_reverse"]
            ) / 2

        if self.metric_config.element_delta:
            computed_metrics["element_delta"] = self.evaluated_mesh.num_elements - self.reference_mesh.num_elements

        return computed_metrics

    def get_fem_metrics(self) -> MetricDict:
        """
        Calculate fem-specific mesh quality metrics, i.e., metrics that evaluate the mesh w.r.t. an underlying PDE.
        Only available if we actually have such a PDE.
        Returns:

        """
        assert self.fem_problem is not None
        fem_metrics = self.fem_problem.get_quality_metrics(self.evaluated_mesh)
        return prefix_keys(fem_metrics, "fem", separator="_")

    def projected_l2_error(self, reverse: bool = False) -> float:
        """
        Computes the relative L2 norm error between the sizing fields of the adaptive
        and reference meshes after projecting the reference field onto the adaptive mesh.

        Args:
            reverse (bool): If True, swaps the roles of the adaptive and reference meshes.

        Returns:
            float: Relative L2 error metric quantifying the difference between the two meshes.
        """
        from src.algorithm.util.amber_util import interpolate_vertex_field

        # Compute vertex-based sizing fields
        sizing_1 = get_sizing_field(self.reference_mesh, mesh_node_type="vertex")
        sizing_2 = get_sizing_field(self.evaluated_mesh, mesh_node_type="vertex")

        # Reverse case: swap reference and adaptive mesh roles
        if reverse:
            sizing_1, sizing_2 = sizing_2, sizing_1
            mesh_1, mesh_2 = self.evaluated_mesh, self.reference_mesh
        else:
            mesh_1, mesh_2 = self.reference_mesh, self.evaluated_mesh

        # Project the sizing field from the vertices of mesh_1 onto those of mesh_2
        sizing_1_projected = interpolate_vertex_field(mesh_1, mesh_2, sizing_1)

        # Compute L2 norm of the difference
        l2_diff = np.linalg.norm(sizing_2 - sizing_1_projected, ord=2)
        l2_ref = np.linalg.norm(sizing_1_projected, ord=2)

        # Normalize by reference field norm with small epsilon to prevent division by zero
        return l2_diff / (l2_ref + 1e-10)

    def pointcloud_distances(self, all_point_metrics: Dict | DictConfig) -> MetricDict:
        """
        Compute pointcloud distances between the reference and evaluated meshes. This includes the Chamfer distance,
        the density-aware Chamfer distance, the exponentiated Chamfer distance, and the Earth Mover's Distance (EMD).
        Args:
            all_point_metrics: Dictionary of names of metrics to include. Has structure
                midpoints: [{cd, dcd, ecd, emd}],
                vertices: [{cd, dcd, ecd, emd}]
            to include the Chamfer distance, density-aware Chamfer distance, exponentiated Chamfer distance, and
            Earth Mover's Distance (EMD) for element midpoints and/or mesh vertices.

        Returns:

        """
        pointcloud_distances = {}
        for scope, point_metrics in all_point_metrics.items():
            if "cd" in point_metrics:  # chamfer distance
                pointcloud_distances[f"cd_{scope}"] = self._chamfer_distance(scope=scope, distance_type="vanilla")
            if "dcd" in point_metrics:  # density-aware chamfer distance
                pointcloud_distances[f"dcd_{scope}"] = self._chamfer_distance(scope=scope, distance_type="density_aware")
            if "ecd" in point_metrics:  # exponentiated chamfer distance
                pointcloud_distances[f"ecd_{scope}"] = self._chamfer_distance(scope=scope, distance_type="exponentiated")
        return pointcloud_distances

    def _chamfer_distance(self, scope: str = "midpoint", distance_type: str = "vanilla") -> float:
        if scope == "midpoint":
            distances1, indices1, distances2, indices2 = self._midpoint_distances_and_indices
        elif scope == "vertex":
            distances1, indices1, distances2, indices2 = self._vertex_distances_and_indices
        else:
            raise ValueError(f"Unknown scope '{scope}'")
        if distance_type == "vanilla":
            distance = 0.5 * (np.mean(distances1) + np.mean(distances2))
        elif distance_type == "density_aware" or distance_type == "exponentiated":
            exp_distances1 = np.exp(-distances1)
            exp_distances2 = np.exp(-distances2)
            if distance_type == "exponentiated":
                distance = 0.5 * (np.mean(1 - exp_distances1) + np.mean(1 - exp_distances2))
            elif distance_type == "density_aware":
                weighted_distances1 = exp_distances1 / np.bincount(indices1).astype(np.float32)[indices1]
                weighted_distances2 = exp_distances2 / np.bincount(indices2).astype(np.float32)[indices2]
                distance = 0.5 * (np.mean(1 - weighted_distances1) + np.mean(1 - weighted_distances2))
            else:
                raise ValueError(f"Unknown distance type '{distance_type}'")
        else:
            raise ValueError(f"Unknown distance type '{distance_type}'")
        return distance

    @cached_property
    def _midpoint_distances_and_indices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        distances1, indices1 = self.reference_mesh.midpoint_tree.query(self.evaluated_mesh.element_midpoints, k=1)
        distances2, indices2 = self.evaluated_mesh.midpoint_tree.query(self.reference_mesh.element_midpoints, k=1)
        return distances1, indices1, distances2, indices2

    @cached_property
    def _vertex_distances_and_indices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        distances1, indices1 = self.reference_mesh.vertex_tree.query(self.evaluated_mesh.vertex_positions, k=1)
        distances2, indices2 = self.evaluated_mesh.vertex_tree.query(self.reference_mesh.vertex_positions, k=1)
        return distances1, indices1, distances2, indices2
