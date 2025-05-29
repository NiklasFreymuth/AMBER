from typing import Union

import numpy as np


def volume_to_edge_length(element_volumes: Union[float, np.ndarray], dim: int):
    """
    Convert the volume of a simplex to the average edge length of that simplex
    Inverse of edge_length_to_volume
    Args:
        element_volumes: Numpy array of arbitrary shape. Will be used to approximate the edge length for this volume,
            which is the average edge length of a simplex with this volume. The modulation is different for 2D and 3D
            meshes:
            2d: sqrt(4 / sqrt(3))
            3d: 12 / sqrt(2) ** (1/3)
        dim: Dimensionality of the underlying mesh

    Returns:

    """
    if dim == 2:
        expert_sizing_field = np.sqrt(4 / np.sqrt(3) * element_volumes)
    elif dim == 3:
        expert_sizing_field = np.power(12 / np.sqrt(2) * element_volumes, 1 / 3)
    else:
        raise ValueError(f"Mesh dimension {dim} not supported")
    return expert_sizing_field


def edge_length_to_volume(sizing_field: np.ndarray, dim: int) -> np.ndarray:
    """
    Get the estimated element size of a new mesh created from the given sizing field.
    Inverse of volume_to_edge_length
    Args:
        sizing_field: Sizing field to use for the element size estimation. Contains one value per element
        dim: Dimension of the mesh. Either 2 or 3

    Returns: Estimated element size of the new mesh per old element

    """
    if dim == 2:
        average_element_size = sizing_field**2 * np.sqrt(3) / 4
    elif dim == 3:
        average_element_size = sizing_field**3 * np.sqrt(2) / 12
    else:
        raise ValueError(f"Dimension {dim} not supported")
    return average_element_size


def get_simplex_volumes_from_indices(positions: np.array, simplex_indices: np.array) -> np.array:
    """
    Computes the volume for an array of simplices.
    Args:
    positions: Array of shape (#points, 3) of (x,y,z) coordinates
    simplex_indices: Array of shape (#simplices, 3) containing point indices that span simplices
    Returns: An array of shape (#simplices,) of volumes for the input simplices
    """
    if positions.shape[-1] == 2:  # 2d case:
        return _get_triangle_areas_from_indices(positions=positions, triangle_indices=simplex_indices)
    elif positions.shape[-1] == 3:  # 3d case:
        return _get_tetrahedron_volumes_from_indices(positions=positions, tetrahedron_indices=simplex_indices)
    else:
        raise ValueError(f"Cannot compute simplex volumes for {positions.shape[-1]} dimensions")


def _get_triangle_areas_from_indices(positions: np.array, triangle_indices: np.array) -> np.array:
    """
    Computes the area for an array of triangles using the triangle-wise formula
    Area = 0.5*| (Xb-Xa)(Yc-Ya)-(Xc-Xa)(Yb-Ya) | where a,b,c are 3 vertices
    for coordinates X and Y
    Args:
        positions: Array of shape (#points, 2) of (x,y) coordinates
        triangle_indices: Array of shape (#triangles, 3) containing point indices that span triangles

    Returns: An array of shape (#triangles,) of areas for the input triangles

    """

    area = np.abs(
        0.5
        * (
            (positions[triangle_indices[:, 1], 0] - positions[triangle_indices[:, 0], 0])
            * (positions[triangle_indices[:, 2], 1] - positions[triangle_indices[:, 0], 1])
            - (positions[triangle_indices[:, 2], 0] - positions[triangle_indices[:, 0], 0])
            * (positions[triangle_indices[:, 1], 1] - positions[triangle_indices[:, 0], 1])
        )
    )
    return area


def _get_tetrahedron_volumes_from_indices(positions: np.array, tetrahedron_indices: np.array) -> np.array:
    """
    Computes the volume for an array of tetrahedra.
    Args:
    positions: Array of shape (#points, 3) of (x,y,z) coordinates
    tetrahedron_indices: Array of shape (#tetrahedra, 4) containing point indices that span tetrahedra
    Returns: An array of shape (#tetrahedra,) of volumes for the input tetrahedra
    """
    # Extract coordinates of tetrahedron vertices
    v0 = positions[tetrahedron_indices[:, 0]]
    v1 = positions[tetrahedron_indices[:, 1]]
    v2 = positions[tetrahedron_indices[:, 2]]
    v3 = positions[tetrahedron_indices[:, 3]]

    # Compute the volume
    volume = np.abs(np.einsum("ij,ij->i", v1 - v0, np.cross(v2 - v0, v3 - v0)) / 6.0)
    return volume
