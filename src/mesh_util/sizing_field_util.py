from typing import Optional

import numpy as np
import torch

from src.helpers.custom_types import MeshNodeType
from src.tasks.domains.mesh_wrapper import MeshWrapper


def get_sizing_field(mesh: MeshWrapper, mesh_node_type: MeshNodeType = "element") -> np.ndarray:
    """
    Computes the sizing field for a given mesh, representing the desired element size in terms of average edge length.

    The sizing field can be computed at the element level (using element-wise average edge lengths) or at the
    vertex level (interpolating element values to vertices).

    Args:
        mesh (MeshWrapper): The input mesh for which the sizing field is computed.
        mesh_node_type (str, optional): Determines whether the sizing field is computed at the
            "element" level (default) or "vertex" level. Valid options:
            - "element": Returns an array of element-wise average edge lengths.
            - "vertex": Projects element-based values to vertex-based values.

    Returns:
        np.ndarray: A numpy array containing the computed sizing field values, either at element or vertex locations.

    Raises:
        ValueError: If an invalid `sizing_field_query_scope` is provided.
    """
    from src.tasks.domains.geometry_util import volume_to_edge_length

    element_volumes = mesh.simplex_volumes
    element_edge_lengths = volume_to_edge_length(element_volumes, mesh.dim())
    if mesh_node_type == "element":
        sizing_field = element_edge_lengths
    elif mesh_node_type == "vertex":
        sizing_field = project_elements_to_vertices(mesh, element_edge_lengths)
    else:
        raise ValueError(f"Sizing field type {mesh_node_type} not recognized")
    return sizing_field


def project_elements_to_vertices(mesh: MeshWrapper, element_values: np.ndarray) -> np.ndarray:
    """
    Projects the provided element-wise values to the nodes of the mesh. This is done by computing the weighted average
    of the element values for each node, where the weights are the volumes of the elements that contain the node.
    This corresponds to solving the FEM for linear elements with the given element values as the right-hand side.
    Transforms the values to torch tensors for an efficient scatter operation and back to numpy arrays to return.

    Args:
        mesh: A MeshWrapper object containing a simplical 2d or 3d mesh
        element_values: Numpy array of shape (num_elements, ) containing the value for each element


    Returns:

    """
    from torch_scatter import scatter_add

    from src.helpers.torch_util import detach

    volumes = mesh.simplex_volumes
    volumes = torch.tensor(np.repeat(volumes, mesh.t.shape[0]))
    element_values = torch.tensor(np.repeat(element_values, mesh.t.shape[0]))
    index = torch.tensor(mesh.t.T.flatten(), dtype=torch.int64)
    vertex_sums = scatter_add(src=element_values * volumes, index=index, dim=0, dim_size=mesh.nvertices)
    vertex_weights = scatter_add(src=volumes, index=index, dim=0, dim_size=mesh.nvertices)
    values_per_vertex = vertex_sums / vertex_weights
    values_per_vertex = detach(values_per_vertex)
    return values_per_vertex


def sizing_field_to_num_elements(
    mesh: MeshWrapper,
    sizing_field: np.ndarray,
    node_type: MeshNodeType,
    pixel_volume: Optional[float] = None,
    gradation: Optional[float] = 1.3,
) -> float:
    if node_type == "pixel":
        return _image_sizing_field_to_num_elements(
            mesh,
            sizing_field,
            pixel_volume=pixel_volume,
            gradation=gradation,
        )
    else:
        return _mesh_sizing_field_to_num_elements(
            mesh,
            sizing_field,
            node_type=node_type,
            gradation=gradation,
        )


def _image_sizing_field_to_num_elements(
    mesh: MeshWrapper,
    sizing_field: np.ndarray,
    pixel_volume: float,
    gradation: Optional[float] = 1.3,
) -> float:
    """
    Estimate the number of elements in a new mesh based on a sizing field living on a regular grid/image.
    Since the GMSH mesh generation is a mystery, we use a heuristic to estimate the number of elements, tending towards
    underestimation where possible.

    Args:
        mesh (MeshWrapper): Mesh object containing geometry and topology data.
        sizing_field (np.ndarray): Array representing the target edge lengths.
        pixel_volume (Optional[np.ndarray]): Required if interpolation_type == "pixels".
        gradation (Optional[float]): Gradation factor for the mesh generation. Default is 1.3. This is essentially the
            aspect ratio between neighboring elements in the new mesh.
            A value of 1.3 means that the longest edge of a new element is at most 1.3 times

    Returns:
        float: Estimated number of elements in the new mesh.
    """
    from src.tasks.domains.geometry_util import edge_length_to_volume

    dim = mesh.dim()

    # pixel-level estimate
    # allow for some extra elements in the estimate via a magic "10", essentially making underestimation more likely.
    element_volumes = edge_length_to_volume(sizing_field, dim)
    return np.sum(pixel_volume / (10 * element_volumes))


def _mesh_sizing_field_to_num_elements(
    mesh: MeshWrapper,
    sizing_field: np.ndarray,
    node_type: Optional[MeshNodeType] = None,
    gradation: Optional[float] = 1.3,
) -> float:
    """
    Estimate the number of elements in a new mesh based on a sizing field.
    Since the GMSH mesh generation is a mystery, we use a heuristic to estimate the number of elements, tending towards
    underestimation where possible.

    Args:
        mesh (MeshWrapper): Mesh object containing geometry and topology data.
        sizing_field (np.ndarray): Array representing the target edge lengths.
        node_type (str): One of {"vertex", "element"}.
        gradation (Optional[float]): Gradation factor for the mesh generation. Default is 1.3. This is essentially the
            aspect ratio between neighboring elements in the new mesh.
            A value of 1.3 means that the longest edge of a new element is at most 1.3 times

    Returns:
        float: Estimated number of elements in the new mesh.
    """
    from src.tasks.domains.geometry_util import edge_length_to_volume

    dim = mesh.dim()

    simplex_volumes = mesh.simplex_volumes

    if node_type == "vertex":
        # "vertex"  underestimates the number of generated elements, especially for very large/sparse sizing fields.

        # We utilize the gradation factor to "smoothen" local outliers in the prediction based on their distance
        # to neighboring vertices.
        edges = mesh.mesh_edges

        pos0 = mesh.vertex_positions[edges[0]]
        pos1 = mesh.vertex_positions[edges[1]]
        edge_distances = np.linalg.norm(pos0 - pos1, axis=1)

        sizing_field = sizing_field.copy()

        # Smooth the sizing field using the gradation factor for 2 passes, i.e., along a 2-neighborhood per vertex
        for _ in range(2):
            sizing0 = sizing_field[edges[0]]
            sizing1 = sizing_field[edges[1]]

            # Forward: enforce sizing1 >= sizing0 * g^(-d / s0)
            n_forward = edge_distances / sizing0
            min_sizing1 = sizing0 * gradation ** (-n_forward)

            # Backward: enforce sizing0 >= sizing1 * g^(-d / s1)
            n_backward = edge_distances / sizing1
            min_sizing0 = sizing1 * gradation ** (-n_backward)

            # Apply max update conservatively
            np.maximum.at(sizing_field, edges[0], min_sizing0)
            np.maximum.at(sizing_field, edges[1], min_sizing1)

        element_sizing_field = sizing_field[mesh.t].mean(axis=0)
        element_predicted_volumes = edge_length_to_volume(element_sizing_field, dim)

        element_density = simplex_volumes / element_predicted_volumes
        num_predicted_elements = element_density.sum()
        if dim == 3:
            # manual correction factor for complex 3d geometries, which tend to place fewer elements than expected
            num_predicted_elements = num_predicted_elements * 0.618
        return num_predicted_elements
    elif node_type == "element":
        predicted_volumes = edge_length_to_volume(sizing_field, dim)
        element_density = simplex_volumes / predicted_volumes
        return element_density.sum()
    else:
        raise ValueError(f"Unsupported interpolation_type: {node_type}")


def smooth_sizing_field(mesh, sizing_field, n_iters=5, device="cpu"):
    # Convert to torch
    sizing_field = torch.tensor(sizing_field, dtype=torch.float32, device=device)
    t = torch.tensor(mesh.t, dtype=torch.int64, device=device)  # shape: (n_nodes_per_elem, n_elements)
    n_vertices = sizing_field.shape[0]

    for _ in range(n_iters):
        # 1. Build list of all vertex-vertex connections via elements
        # Each vertex in each element connects to the other vertices
        neighbors = t.flatten()  # all vertex indices appearing in elements
        repeated = t.repeat(t.shape[0], 1).flatten()  # for each node, all others in the same element

        # Remove self-connections
        mask = neighbors != repeated
        i = neighbors[mask]  # target vertex index
        j = repeated[mask]  # source vertex index

        # 2. Scatter-add sizing field contributions
        neighbor_sum = torch.zeros_like(sizing_field)
        neighbor_count = torch.zeros_like(sizing_field)

        neighbor_sum = neighbor_sum.scatter_add(0, i, sizing_field[j])
        neighbor_count = neighbor_count.scatter_add(0, i, torch.ones_like(j, dtype=sizing_field.dtype))

        # 3. Include self in the average
        neighbor_sum += sizing_field
        neighbor_count += 1

        sizing_field = neighbor_sum / neighbor_count

    return sizing_field.cpu().numpy()


def scale_sizing_field_to_budget(
    sizing_field: np.ndarray,
    mesh: MeshWrapper,
    max_elements: float,
    node_type: MeshNodeType,
    pixel_volume: Optional[float] = None,
    tolerance: float = 0.99,
    max_iter: int = 20,
) -> np.ndarray:
    """
    Scale the sizing field to reduce the estimated number of mesh elements below a given budget using binary search.
    A lower sizing field corresponds to more elements, while a higher sizing field corresponds to fewer elements.

    Args:
        sizing_field (np.ndarray): Original sizing field.
        mesh (MeshWrapper): Mesh used for estimation.
        node_type (MeshNodeType): One of {"vertex", "element", "pixels"}.
        max_elements (float): Maximum allowable number of mesh elements.
        pixel_volume (Optional[np.ndarray]): Required if node_type == "pixels".
        tolerance (float): Allowed deviation from max_elements in percent.
        max_iter (int): Maximum iterations for binary search.

    Returns:
        np.ndarray: Scaled sizing field that fits the mesh element budget.
    """
    lower, upper = 0.0, 1.0
    inverse_scaling_factor = 1.0

    for _ in range(max_iter):
        inverse_scaling_factor = (lower + upper) / 2
        current_elements = sizing_field_to_num_elements(
            mesh,
            sizing_field * 1 / inverse_scaling_factor,
            node_type=node_type,
            pixel_volume=pixel_volume,
        )
        if current_elements > max_elements:
            upper = inverse_scaling_factor
        elif current_elements < max_elements * tolerance:
            lower = inverse_scaling_factor
        else:
            break

    return sizing_field / inverse_scaling_factor
