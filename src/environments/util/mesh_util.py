from functools import partial, update_wrapper
from typing import Any, Dict, List, Union

import numpy as np
from skfem import Mesh


def get_element_midpoints(mesh: Mesh, transpose: bool = True) -> np.array:
    """
    Get the midpoint of each element
    Args:
        mesh: The mesh as a skfem.Mesh object
        transpose: Whether to transpose the result. If True
            the result will be of shape (num_elements, 2), if False, it will be of shape (2, num_elements). Defaults
            to True.

    Returns: Array of shape (num_elements, 2)/(2, num_elements) containing the midpoint of each element

    """
    midpoints = np.mean(mesh.p[:, mesh.t], axis=1)
    if transpose:
        return midpoints.T
    else:
        return midpoints


def get_aggregation_per_element(
        solution: np.array,
        element_indices: np.array,
        aggregation_function_str: str = "mean",
) -> np.array:
    """
    get aggregation of solution per element from solution per vertex by adding all spanning vertices for each element

    Args:
        solution: Error estimate per element of shape (num_elements, solution_dimension)
        element_indices: Elements of the mesh. Array of shape (num_elements, vertices_per_element),
        where vertices_per_element is 3 triangular meshes
        aggregation_function_str: The aggregation function to use. Can be 'mean', 'std', 'min', 'max', 'median'
    Returns: An array of shape (num_elements, ) containing the solution per element

    """
    if aggregation_function_str == "mean":
        solution_per_element = solution[element_indices].mean(axis=1)
    elif aggregation_function_str == "std":
        solution_per_element = solution[element_indices].std(axis=1)
    elif aggregation_function_str == "min":
        solution_per_element = solution[element_indices].min(axis=1)
    elif aggregation_function_str == "max":
        solution_per_element = solution[element_indices].max(axis=1)
    elif aggregation_function_str == "median":
        solution_per_element = np.median(solution[element_indices], axis=1)
    else:
        raise ValueError(f"Aggregation function {aggregation_function_str} not supported")
    return solution_per_element


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


def filter_included_fields(dictionary: Dict[Union[str, int], Any]) -> List[str]:
    """
    A helper function to filter out the fields for features that are not included in the config.
    Args:
        dictionary: A dictionary containing the fields to filter. The values are booleans indicating whether to include
            the field.

    Returns:

    """
    return [feature_name for feature_name, include_feature in dictionary.items() if include_feature]


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func
