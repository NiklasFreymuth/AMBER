from typing import List, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data, HeteroData

from src.algorithms.amber.mesh_wrapper import MeshWrapper
from src.environments.problems import AbstractFiniteElementProblem
from util.function import save_concatenate


def mesh_to_graph(wrapped_mesh: MeshWrapper, labels: np.ndarray, element_feature_names: List[str],
                  edge_feature_names: List[str], fem_problem: Optional[AbstractFiniteElementProblem],
                  device: str) -> Union[Data, HeteroData]:
    """
    Generates an observation graph from a finite element problem and a sizing field.
    The graph is used as input for the GNN-based supervised learning algorithm.
    Args:
        wrapped_mesh: CachedMeshWrapper containing the mesh data as required for, e.g., the mesh connectivity and
            element midpoints
        fem_problem: FEM problem containing problem-specific information, such as boundary conditions and material
            May be None.
        labels: Label for the supervised learning algorithm. Will usually be the sizing field.
        element_feature_names: Names of the features to use for the element nodes
        edge_feature_names: Names of the features to use for the edges between the elements in the observation graph
        device: The (accelerator) device to use for the graph data
    Returns:

    """
    element_features = get_mesh_element_features(wrapped_mesh,
                                                 fem_problem=fem_problem,
                                                 element_feature_names=element_feature_names)
    edge_attr, edge_index = _get_mesh_element2element_features(wrapped_mesh, edge_feature_names)

    graph_dict = {
        "x": element_features,
        "y": torch.tensor(labels, dtype=torch.float32),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }
    graph = Data(**graph_dict)
    graph = graph.to(device)
    return graph


def _get_mesh_element2element_features(wrapped_mesh: MeshWrapper, edge_feature_names: List[str]):
    src_nodes = np.concatenate(
        (
            wrapped_mesh.element_neighbors[0],
            wrapped_mesh.element_neighbors[1],
            np.arange(wrapped_mesh.num_elements),
        ),
        axis=0,
    )
    dest_nodes = np.concatenate(
        (
            wrapped_mesh.element_neighbors[1],
            wrapped_mesh.element_neighbors[0],
            np.arange(wrapped_mesh.num_elements),
        ),
        axis=0,
    )
    edge_features = []
    if "distance_vector" in edge_feature_names:
        distance_vectors = wrapped_mesh.element_midpoints[dest_nodes] - wrapped_mesh.element_midpoints[src_nodes]
        edge_features.extend(list(distance_vectors.T))
    if "euclidean_distance" in edge_feature_names:
        euclidean_distances = np.linalg.norm(
            wrapped_mesh.element_midpoints[dest_nodes] - wrapped_mesh.element_midpoints[src_nodes],
            axis=1,
        )
        edge_features.append(euclidean_distances)
    edge_index = torch.tensor(np.vstack((src_nodes, dest_nodes))).long()
    edge_features = np.array(edge_features).T
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    return edge_attr, edge_index


def get_mesh_element_features(wrapped_mesh: MeshWrapper,
                              fem_problem: Optional[AbstractFiniteElementProblem],
                              element_feature_names: List[str]):
    """
    Extracts the general element features for a given mesh and fem problem. These features are used as node features
    for the graph representation in the supervised learning algorithm.
    Args:
        wrapped_mesh:
        fem_problem:
        element_feature_names:

    Returns:

    """
    from src.environments.util.mesh_util import get_aggregation_per_element

    general_element_features = []
    if "x_position" in element_feature_names:
        general_element_features.append(wrapped_mesh.element_midpoints[:, 0])
    if "y_position" in element_feature_names:
        general_element_features.append(wrapped_mesh.element_midpoints[:, 1])
    if "volume" in element_feature_names:
        general_element_features.append(wrapped_mesh.element_volumes)

    if any("solution" in feature_name for feature_name in element_feature_names):
        assert fem_problem is not None, "FEM problem must be provided to extract solution features"
        solution = fem_problem.calculate_solution(mesh=wrapped_mesh)
        if "solution_mean" in element_feature_names:
            solution_mean = get_aggregation_per_element(
                solution=solution,
                element_indices=wrapped_mesh.element_indices,
                aggregation_function_str="mean",
            )
            for solution_dimension in range(solution_mean.shape[1]):
                general_element_features.append(solution_mean[:, solution_dimension])
        if "solution_std" in element_feature_names:
            solution_std = get_aggregation_per_element(
                solution=solution,
                element_indices=wrapped_mesh.element_indices,
                aggregation_function_str="std",
            )
            for solution_dimension in range(solution_std.shape[1]):
                general_element_features.append(solution_std[:, solution_dimension])

    # Convert general_element_features to a NumPy array if not empty, else set it as None
    general_element_features = np.array(general_element_features).T if general_element_features else None

    if fem_problem is not None:
        problem_element_features = fem_problem.element_features(mesh=wrapped_mesh.mesh)
        problem_element_features = problem_element_features if problem_element_features is not None else None
        # Use save_concatenate to handle both empty and non-empty cases
        element_features = save_concatenate([general_element_features, problem_element_features], axis=1)
    else:
        element_features = general_element_features

    if element_features is None:
        element_features = torch.zeros((wrapped_mesh.num_elements, 0), dtype=torch.float32)
    else:
        element_features = torch.tensor(element_features, dtype=torch.float32)

    return element_features
