from typing import Optional

import numpy as np
import torch
from skfem import Basis, ElementTetP1, ElementTriP1

from src.helpers.custom_types import MeshNodeType
from src.mesh_util.sizing_field_util import get_sizing_field
from src.tasks.domains.mesh_wrapper import MeshWrapper


def interpolate_vertex_field(from_mesh: MeshWrapper, to_mesh: MeshWrapper, from_scalars: np.ndarray) -> np.ndarray:
    """
    Projects a piecewise linear field living on the vertices of one mesh to another mesh.
        This is done by evaluating the linear basis of the first mesh on the vertex positions of the second mesh.
    Vertices of the second mesh that lie outside the first mesh are set to the value of the nearest fine vertex.

    Args:
        to_mesh: A MeshWrapper object.
        from_mesh: Another MeshWrapper object
        from_scalars: Array of shape (#from_mesh.vertices, ) containing the values to build a linear basis on the
          fine_mesh from.


    Returns: A numpy array of shape (#to_mesh.vertices, ) containing the projected values of the first mesh on the
        second mesh.

    """
    probes = get_vertex_probes(from_mesh, to_mesh.p)
    # Matrix of shape (#to_mesh.vertices, #from_mesh.vertices) that maps the values of the first mesh to the second
    # Has probes.sum(axis=1)=1 everywhere, i.e., interpolates the values of the first mesh to the second
    to_mesh_scalars = probes @ from_scalars
    return to_mesh_scalars


def get_vertex_probes(probed_mesh: MeshWrapper, query_points: np.ndarray) -> np.ndarray:
    """
    Assign a bunch of query_points to the vertices of a mesh. This is done by finding the element that contains each
    query point, and then assigning the query point to the vertices of this element. The probes are then the convex
    combination of the vertices of the element that contains the query point, where the weight of each vertex
    is (similar to) the barycentric coordinate of the query point in the element.

    If a query_point is not in any element, it is assigned to the nearest vertex of the mesh.

    Args:
        probed_mesh: The mesh *from* which to calculate the probes. I.e., the vertices for this mesh are found
        query_points: Points to query in the element_finder/mapping of the probed_mesh. Has shape (dim, #query_points)

    Returns: A sparse matrix of shape (#query_points, probed_mesh.num_vertices) that maps the query points to mesh
        vertices. Has .sum(axis=1)=1 everywhere.

    """
    from src.mesh_util.mesh_util import probes_from_elements

    scalar_basis = Basis(probed_mesh.mesh, ElementTriP1() if probed_mesh.dim() == 2 else ElementTetP1())
    corresponding_elements = scalar_basis.mesh.element_finder()(*query_points)
    probes = probes_from_elements(scalar_basis, query_points, corresponding_elements)  # coo-matrix

    invalid_elements = corresponding_elements == -1
    # fallback for queries that are not in any fine element, which may happen for, e.g., concave geometries
    # Here, we set assign each point that is not in a fine element fully to its nearest vertex in the fine mesh.
    # I.e., usually, each point is defined as a convex combination of the vertices of the element it is in, but here,
    # The point is directly projected to the nearest vertex.
    if invalid_elements.any():
        vertices_per_element = probed_mesh.t.shape[0]
        fine_tree = probed_mesh.vertex_tree
        _, candidate_indices = fine_tree.query(query_points[:, invalid_elements].T, k=1)
        candidate_indices = candidate_indices.astype(np.int64)
        # to_mesh_scalars[invalid_elements] = from_scalars[candidate_indices]
        probes.data.reshape(vertices_per_element, -1).T[invalid_elements] = 0
        probes.data.reshape(vertices_per_element, -1).T[invalid_elements, 0] = 1
        probes.col.reshape(vertices_per_element, -1).T[invalid_elements] = 0
        probes.col.reshape(vertices_per_element, -1).T[invalid_elements, 0] = candidate_indices
    return probes


def get_scatter_reduce(interpolation_type: str) -> callable:
    if interpolation_type == "mean":
        from torch_scatter import scatter_mean

        scatter_reduce = scatter_mean

    elif interpolation_type == "max":
        from torch_scatter import scatter_max

        scatter_reduce = lambda *args, **kwargs: scatter_max(*args, **kwargs)[0]
    elif interpolation_type == "min":
        from torch_scatter import scatter_min

        scatter_reduce = lambda *args, **kwargs: scatter_min(*args, **kwargs)[0]
    elif interpolation_type == "rmse":
        from torch_scatter import scatter_mean

        scatter_reduce = lambda src, *args, **kwargs: torch.sqrt(scatter_mean(src**2, *args, **kwargs))
    else:
        raise ValueError(f"Unknown interpolation type '{interpolation_type}'")
    return scatter_reduce


def get_reconstructed_mesh(
    reference_mesh: MeshWrapper,
    mesh_node_type: MeshNodeType = "vertex",
    gmsh_kwargs: Optional[dict] = None,
) -> MeshWrapper:
    """
    Calculate a reference mesh from an expert mesh. This is done by calculating a sizing field on the expert mesh,
    and then building a new mesh from this sizing field. This is essentially an upper bound on the mesh quality
    that we can achieve when we want to reconstruct a mesh from a sizing field.
    Args:
        reference_mesh:
        mesh_node_type: The type of sizing field to use. Can be "element" or "vertex"
        gmsh_kwargs: Additional keyword arguments for the Gmsh mesh generation algorithm

    Returns: A mesh that is reconstructed from the sizing field of the expert mesh

    """
    from src.tasks.domains.update_mesh import update_mesh

    sizing_field = get_sizing_field(mesh=reference_mesh, mesh_node_type=mesh_node_type)
    reconstructed_mesh = update_mesh(old_mesh=reference_mesh.mesh, sizing_field=sizing_field, gmsh_kwargs=gmsh_kwargs)
    return reconstructed_mesh
