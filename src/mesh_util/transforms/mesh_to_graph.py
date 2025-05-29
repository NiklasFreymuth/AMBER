from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from src.helpers.custom_types import MeshNodeType
from src.helpers.qol import safe_concatenate
from src.tasks.domains.mesh_wrapper import MeshWrapper
from src.tasks.features.feature_provider import FeatureProvider


def mesh_to_graph(
    wrapped_mesh: MeshWrapper,
    node_feature_names: List[str],
    edge_feature_names: List[str],
    feature_provider: Optional[FeatureProvider],
    node_type: MeshNodeType = "element",
    add_self_edges: bool = True,
) -> Data:
    """
    Generates an observation graph from a finite element problem and a sizing field.
    The graph is used as input for the GNN-based supervised learning algorithm.
    Args:
        wrapped_mesh: MeshWrapper containing the mesh data as required for, e.g., the mesh connectivity and
            element midpoints
        feature_provider: Class containing problem-specific information, such as boundary conditions and material for a FEM,
            or inlet position for the molding task.
            May be None if a task has no features.
        node_feature_names: Names of the features to use for the element nodes
        edge_feature_names: Names of the features to use for the edges between the elements in the observation graph
        node_type: The type of the nodes in the graph. Either "element" for a graph over mesh elements, or
                "vertex" for a graph over mesh vertices
    Returns:

    """
    if node_type == "element":
        node_features = get_mesh_element_features(wrapped_mesh, feature_provider=feature_provider, node_feature_names=node_feature_names)
        edge_attr, edge_index = get_mesh_element_edges(wrapped_mesh, edge_feature_names=edge_feature_names, add_self_edges=add_self_edges)
    elif node_type == "vertex":
        node_features = get_mesh_vertex_features(wrapped_mesh, feature_provider=feature_provider, node_feature_names=node_feature_names)
        edge_attr, edge_index = get_mesh_vertex_edges(wrapped_mesh, edge_feature_names=edge_feature_names, add_self_edges=add_self_edges)

    else:
        raise ValueError(f"Node type {node_type=} not supported")

    graph_dict = {
        "x": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }

    graph = Data(**graph_dict)
    return graph


def get_mesh_element_edges(
    wrapped_mesh: MeshWrapper, edge_feature_names: List[str], add_self_edges: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get edges for a mesh. The edge features are computed based on the edge_feature_names.
    Args:
        wrapped_mesh:
        edge_feature_names:
        add_self_edges:

    Returns:

    """
    node_neighbors = torch.tensor(wrapped_mesh.element_neighbors, dtype=torch.long)
    node_positions = torch.tensor(wrapped_mesh.element_midpoints, dtype=torch.float32)

    src_nodes = torch.cat([node_neighbors[0], node_neighbors[1]], dim=0)
    dest_nodes = torch.cat([node_neighbors[1], node_neighbors[0]], dim=0)

    if add_self_edges:
        num_nodes = wrapped_mesh.num_elements
        src_nodes = torch.cat([src_nodes, torch.arange(num_nodes)], dim=0)
        dest_nodes = torch.cat([dest_nodes, torch.arange(num_nodes)], dim=0)

    edge_features = []
    if "distance_vector" in edge_feature_names:
        distance_vectors = node_positions[dest_nodes] - node_positions[src_nodes]
        edge_features.extend(list(distance_vectors.T))

    if "euclidean_distance" in edge_feature_names:
        euclidean_distances = torch.norm(node_positions[dest_nodes] - node_positions[src_nodes], dim=1)
        edge_features.append(euclidean_distances)

    edge_index = torch.vstack((src_nodes, dest_nodes)).long()
    edge_attr = torch.stack(edge_features, dim=1)  # shape: [num_edges, num_features]

    return edge_attr, edge_index


def get_mesh_vertex_edges(wrapped_mesh: MeshWrapper, edge_feature_names: List[str], add_self_edges: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get edges for a mesh. The edge features are computed based on the edge_feature_names.
    Args:
        wrapped_mesh:
        edge_feature_names:
        add_self_edges:

    Returns:

    """
    node_neighbors = torch.tensor(wrapped_mesh.mesh_edges, dtype=torch.long)
    node_positions = torch.tensor(wrapped_mesh.vertex_positions, dtype=torch.float32)

    src_nodes = torch.cat([node_neighbors[0], node_neighbors[1]], dim=0)
    dest_nodes = torch.cat([node_neighbors[1], node_neighbors[0]], dim=0)

    if add_self_edges:
        num_nodes = wrapped_mesh.num_vertices
        src_nodes = torch.cat([src_nodes, torch.arange(num_nodes)], dim=0)
        dest_nodes = torch.cat([dest_nodes, torch.arange(num_nodes)], dim=0)

    edge_features = []
    if "distance_vector" in edge_feature_names:
        distance_vectors = node_positions[dest_nodes] - node_positions[src_nodes]
        edge_features.extend(list(distance_vectors.T))

    if "euclidean_distance" in edge_feature_names:
        euclidean_distances = torch.norm(node_positions[dest_nodes] - node_positions[src_nodes], dim=1)
        edge_features.append(euclidean_distances)

    if "edge_curvature" in edge_feature_names:
        boundary_edge_curvatures = wrapped_mesh.boundary_edge_curvatures
        edge_curvatures = np.zeros(wrapped_mesh.mesh_edges.shape[1])
        edge_curvatures[wrapped_mesh.boundary_edges] = boundary_edge_curvatures

        # repeat for both directions
        edge_curvatures = np.concatenate((edge_curvatures, edge_curvatures), axis=0)

        # add 0s for self edges if add_self_edges, as there is no curvature between a node and itself
        if add_self_edges:
            edge_curvatures = np.concatenate((edge_curvatures, np.zeros(wrapped_mesh.num_vertices)), axis=0)
        edge_curvatures = torch.tensor(edge_curvatures, dtype=torch.float32)
        edge_features.append(edge_curvatures)

    edge_index = torch.vstack((src_nodes, dest_nodes)).long()
    edge_attr = torch.stack(edge_features, dim=1)  # shape: [num_edges, num_features]

    return edge_attr, edge_index


def get_mesh_element_features(wrapped_mesh: MeshWrapper, feature_provider: Optional[FeatureProvider], node_feature_names: List[str]) -> torch.Tensor:
    """
    Extracts the general element features for a given mesh and fem problem. These features are used as node features
    for the graph representation in the supervised learning algorithm.
    Args:
        wrapped_mesh:
        feature_provider:
        node_feature_names:

    Returns:

    """
    general_element_features = []
    if "x_position" in node_feature_names:
        general_element_features.append(wrapped_mesh.element_midpoints[:, 0])
    if "y_position" in node_feature_names:
        general_element_features.append(wrapped_mesh.element_midpoints[:, 1])
    if "z_position" in node_feature_names:
        assert wrapped_mesh.dim() == 3, "z_position is only available for 3D meshes"
        general_element_features.append(wrapped_mesh.element_midpoints[:, 2])
    if "sizing_field" in node_feature_names:
        from src.mesh_util.sizing_field_util import get_sizing_field

        general_element_features.append(get_sizing_field(mesh=wrapped_mesh, mesh_node_type="element"))

    general_element_features = np.array(general_element_features).T if general_element_features else None

    # add solution and fem-specific features
    if feature_provider is not None:
        element_fem_features = feature_provider.get_element_features(wrapped_mesh=wrapped_mesh)
        element_features = safe_concatenate([general_element_features, element_fem_features], axis=1)
    else:
        element_features = general_element_features

    # Convert to torch
    if element_features is None:
        # empty tensor
        element_features = torch.zeros((wrapped_mesh.num_elements, 0), dtype=torch.float32)
    else:
        element_features = torch.tensor(element_features, dtype=torch.float32)

    return element_features


def get_mesh_vertex_features(wrapped_mesh: MeshWrapper, feature_provider: Optional[FeatureProvider], node_feature_names: List[str]):
    """
    Extracts the general vertex features for a given mesh and fem problem. These features are used as node features
    for the graph representation in the supervised learning algorithm.
    Args:
        wrapped_mesh:
        feature_provider:
        node_feature_names:

    Returns:

    """

    general_vertex_features = []
    if "x_position" in node_feature_names:
        general_vertex_features.append(wrapped_mesh.p[0])
    if "y_position" in node_feature_names:
        general_vertex_features.append(wrapped_mesh.p[1])
    if "z_position" in node_feature_names:
        assert wrapped_mesh.dim() == 3, "z_position is only available for 3D meshes"
        general_vertex_features.append(wrapped_mesh.p[2])
    if "degree" in node_feature_names:
        general_vertex_features.append(np.unique(wrapped_mesh.mesh_edges, return_counts=True)[1])
    if "sizing_field" in node_feature_names:
        from src.mesh_util.sizing_field_util import get_sizing_field

        general_vertex_features.append(get_sizing_field(mesh=wrapped_mesh, mesh_node_type="vertex"))

    general_vertex_features = np.array(general_vertex_features).T if general_vertex_features else None

    # add solution and fem-specific features
    if feature_provider is not None:
        problem_vertex_features = feature_provider.get_vertex_features(wrapped_mesh=wrapped_mesh)
        vertex_features = safe_concatenate([general_vertex_features, problem_vertex_features], axis=1)
    else:
        vertex_features = general_vertex_features

    # Convert to torch
    if vertex_features is None:
        # empty tensor
        vertex_features = torch.zeros((wrapped_mesh.num_elements, 0), dtype=torch.float32)
    else:
        vertex_features = torch.tensor(vertex_features, dtype=torch.float32)

    return vertex_features


def get_inter_graph_edges(
    src_mesh: MeshWrapper, dest_mesh: MeshWrapper, node_type: str, edge_feature_names: List[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get symmetric edges from one graph to the other. The edge features are computed based on the edge_feature_names.
    Already includes index offsets for the second graph.
    Args:
        src_mesh:
        dest_mesh:  Mesh to compute the edges "to". Usually the smaller of the two meshes
        node_type:
        edge_feature_names:

    Returns:

    """
    if node_type == "element":
        index_offset = src_mesh.num_elements
        src_node_indices = np.arange(src_mesh.num_elements)
        src_node_positions = src_mesh.element_midpoints
        # define elem1 of from_mesh and elem2 of to_mesh as neighbors iff elem1.midpoint is in elem2.
        dest_node_indices = dest_mesh.find_closest_elements(src_node_positions)
        dest_node_positions = dest_mesh.element_midpoints[dest_node_indices]
        dest_node_indices = dest_node_indices + index_offset

    elif node_type == "vertex":
        index_offset = src_mesh.num_vertices
        src_node_indices = np.arange(src_mesh.num_vertices)
        src_node_positions = src_mesh.vertex_positions

        # define vertex1 of from_mesh and vertex2 of to_mesh as neighbors iff vertex1.positions is the closest to
        # vertex2.midpoint for all vertices in dest_mesh
        dest_node_indices = dest_mesh.vertex_tree.query(src_node_positions, k=1)[1]
        dest_node_positions = dest_mesh.vertex_positions[dest_node_indices]
        dest_node_indices = dest_node_indices + index_offset

    else:
        raise ValueError(f"Node type {node_type=} not supported")

    positions1 = np.concatenate((src_node_positions, dest_node_positions), axis=0)
    positions2 = np.concatenate((dest_node_positions, src_node_positions), axis=0)

    indices1 = np.concatenate((src_node_indices, dest_node_indices), axis=0)
    indices2 = np.concatenate((dest_node_indices, src_node_indices), axis=0)

    edge_features = []
    if "distance_vector" in edge_feature_names:
        distance_vectors = positions1 - positions2
        edge_features.extend(list(distance_vectors.T))

    if "euclidean_distance" in edge_feature_names:
        euclidean_distances = np.linalg.norm(positions1 - positions2, axis=1)
        edge_features.append(euclidean_distances)

    if node_type == "vertex":
        if "edge_curvature" in edge_feature_names:
            # no curvature features between two different meshes. Leave empty.
            edge_features.append(np.zeros(len(positions1)))

    edge_index = torch.tensor(np.vstack((indices1, indices2))).long()
    edge_features = np.array(edge_features).T
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)

    return edge_attr, edge_index
