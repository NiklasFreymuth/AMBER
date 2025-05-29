"""
Utility functions for the GraphMesh baseline, which creates a boundary graph for each vertex in the mesh.
"""
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import Data

from src.tasks.domains.mesh_wrapper import MeshWrapper
from src.tasks.features.feature_provider import FeatureProvider


def get_boundary_vertex_graphs(
    wrapped_mesh: MeshWrapper, feature_provider: Optional[FeatureProvider], boundary_graph_feature_names: List[str]
) -> List[Data]:
    """
    Generates boundary vertex graphs for each vertex in the mesh. Each graph contains the boundary polygon with
    mean value coordinates as features.

    Args:
        wrapped_mesh: MeshWrapper containing the mesh data
        feature_provider: Class containing problem-specific information
        boundary_graph_feature_names: Names of the features to use for the vertex nodes

    Returns:
        List[Data]: List of graph objects, one for each mesh vertex
    """
    # Get vertex features with special focus on boundary polygon information
    boundary_features = get_mesh_vertex_features_for_boundary_polygon(
        wrapped_mesh=wrapped_mesh, feature_provider=feature_provider, boundary_graph_feature_names=boundary_graph_feature_names
    )

    # Create individual graphs for each mesh vertex
    return _create_vertex_boundary_graphs(wrapped_mesh, boundary_features, boundary_graph_feature_names)


def _create_vertex_boundary_graphs(
    wrapped_mesh: MeshWrapper, boundary_features: List[torch.Tensor], boundary_graph_feature_names: List[str]
) -> List[Data]:
    """
    Creates individual graphs for each mesh vertex, where each graph represents
    the boundary polygon with mean value coordinates as features.

    Args:
        wrapped_mesh: MeshWrapper containing the mesh data and boundary polygon
        boundary_features: Features calculated for mesh vertices including MVC
        boundary_graph_feature_names: Names of the features used

    Returns:
        List[Data]: List of graph objects, one for each mesh vertex
    """
    # Check if boundary polygon exists
    has_boundary_polygon = hasattr(wrapped_mesh.mesh, "boundary_polygon") and wrapped_mesh.mesh.boundary_polygon is not None

    if not has_boundary_polygon:
        return None

    # Get boundary polygon vertices
    polygon_vertices = wrapped_mesh.mesh.boundary_polygon["nodes"]
    polygon_indices = wrapped_mesh.mesh.boundary_polygon["indices"]
    num_polygon_vertices = len(polygon_vertices)

    # Create edges connecting consecutive vertices in the polygon (forming a loop)
    polygon_edges = [(i, (i + 1) % num_polygon_vertices) for i in range(num_polygon_vertices)]
    polygon_edge_index = torch.tensor(polygon_edges, dtype=torch.long).t().contiguous()

    # Check if MVC features are in the node_feature_names
    if "boundary_mean_value_coordinates" not in boundary_graph_feature_names:
        return None

    # Create a graph for each mesh vertex
    vertex_graphs = []
    mvc_weights, hop_distances, spatial_distances, problem_vertex_features = boundary_features
    for vertex_idx in range(wrapped_mesh.num_vertices):
        # Extract MVC weights for this vertex (these are the features for boundary polygon vertices)
        mvc_w = mvc_weights[vertex_idx]
        hop_dist = hop_distances[vertex_idx]
        spatial_dist = spatial_distances[vertex_idx]
        fem_params = problem_vertex_features[polygon_indices]
        fem_params = np.concatenate([fem_params, np.repeat(problem_vertex_features[vertex_idx][None, :], num_polygon_vertices, axis=0)], axis=1)

        x_feat = torch.tensor(np.column_stack([mvc_w, hop_dist, spatial_dist, fem_params]), dtype=torch.float32)

        # Create a Data object for this vertex
        vertex_data = Data(x=x_feat, edge_index=polygon_edge_index)
        vertex_graphs.append(vertex_data)

    return vertex_graphs


def get_mesh_vertex_features_for_boundary_polygon(
    wrapped_mesh: MeshWrapper, feature_provider: Optional[FeatureProvider], boundary_graph_feature_names: List[str]
) -> List[torch.Tensor]:
    """
    Extracts vertex features with special focus on boundary polygon information. These features are used as node features
    for the graph representation in the supervised learning algorithm.
    Args:
        wrapped_mesh: MeshWrapper containing the mesh data
        feature_provider: Class containing problem-specific information
        boundary_graph_feature_names: Names of the features to use for the vertex nodes

    Returns:
        Tensor containing the vertex features
    """
    # Check if boundary polygon exists
    has_boundary_polygon = hasattr(wrapped_mesh.mesh, "boundary_polygon") and wrapped_mesh.mesh.boundary_polygon is not None

    if not has_boundary_polygon:
        raise ValueError("This function requires a mesh with a boundary polygon defined")

    # Get boundary polygon vertices and mesh vertex positions
    boundary_polygon_vertices = np.array(wrapped_mesh.mesh.boundary_polygon["nodes"])
    vertex_positions = wrapped_mesh.vertex_positions
    num_vertices = wrapped_mesh.num_vertices
    num_boundary_vertices = len(boundary_polygon_vertices)

    mvc_weights = None
    hop_distances = None
    spatial_distances = None
    problem_vertex_features = None

    # -----------------------------
    # Boundary Distance Calculation
    # -----------------------------
    if "boundary_hop_distance" in boundary_graph_feature_names:
        from scipy.spatial import KDTree as ScipyKDTree

        boundary_kdtree = ScipyKDTree(boundary_polygon_vertices)
        # Find nearest polygon vertex index for each mesh vertex
        _, nearest_indices = boundary_kdtree.query(vertex_positions, k=1)
        hop_distances = np.empty((num_vertices, num_boundary_vertices), dtype=int)
        for mesh_idx in range(num_vertices):
            nearest = nearest_indices[mesh_idx]
            for j in range(num_boundary_vertices):
                d = abs(j - nearest)
                hop_distances[mesh_idx, j] = min(d, num_boundary_vertices - d)

    if "boundary_spatial_distance" in boundary_graph_feature_names:
        # Compute spatial distances: (n_vertices, n_boundary_vertices)
        spatial_distances = np.linalg.norm(vertex_positions[:, None, :] - boundary_polygon_vertices[None, :, :], axis=2)

    # -----------------------------
    # Mean Value Coordinates
    # -----------------------------
    if "boundary_mean_value_coordinates" in boundary_graph_feature_names:
        # Vectorized MVC calculation
        mvc_weights = compute_mvc_weights_vectorized(vertex_positions, boundary_polygon_vertices)

    # Add problem-specific features if provided
    if feature_provider is not None:
        problem_vertex_features = feature_provider.get_vertex_features(wrapped_mesh=wrapped_mesh)

    return mvc_weights, hop_distances, spatial_distances, problem_vertex_features


def compute_mvc_weights_vectorized(points, polygon_vertices):
    epsilon = 1e-12
    num_points = points.shape[0]
    num_vertices = polygon_vertices.shape[0]

    # Compute vectors and distances: shape=(num_points, num_vertices, dim) and (num_points, num_vertices)
    vectors = polygon_vertices[None, :, :] - points[:, None, :]
    distances = np.linalg.norm(vectors, axis=2)

    weights = np.zeros((num_points, num_vertices))

    # Handle points close to a vertex: assign one-hot weights
    too_close = distances < epsilon
    special = np.any(too_close, axis=1)
    for idx in np.where(special)[0]:
        vertex = np.argmax(too_close[idx])
        weights[idx, :] = 0
        weights[idx, vertex] = 1.0

    # Process regular points vectorized
    regular = ~special
    if np.any(regular):
        vec = vectors[regular]  # shape: (n, num_vertices, dim)
        dists = distances[regular]  # shape: (n, num_vertices)

        # Roll to get previous vertex for each point
        vec_prev = np.roll(vec, shift=1, axis=1)
        dists_prev = np.roll(dists, shift=1, axis=1)

        # Compute angles between consecutive vectors
        dot = np.sum(vec_prev * vec, axis=2)
        cos_theta = dot / (dists_prev * dists + epsilon)
        theta = np.arccos(cos_theta)

        # Compute unnormalized weights
        w = np.tan(theta / 2.0) / (dists + epsilon)
        # Normalize weights per point
        w_sum = np.sum(w, axis=1, keepdims=True) + epsilon
        weights[regular] = w / w_sum

    return weights
