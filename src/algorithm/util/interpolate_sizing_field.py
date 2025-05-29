from typing import get_args

import numpy as np
from skfem import Basis, ElementTetP1, ElementTriP1

from src.algorithm.util.amber_util import get_scatter_reduce, interpolate_vertex_field
from src.helpers.custom_types import SizingField, SizingFieldInterpolationType
from src.mesh_util.mesh_util import get_midpoint_correspondences
from src.mesh_util.sizing_field_util import get_sizing_field
from src.tasks.domains.mesh_wrapper import MeshWrapper


def interpolate_sizing_field(
    queried_mesh: MeshWrapper,
    fine_mesh: MeshWrapper,
    sizing_field_interpolation_type: SizingFieldInterpolationType,
) -> SizingField:
    """
    Interpolates the sizing field from fine expert mesh to the coarse mesh. Uses the volumes of the fine mesh elements
    as (a basis for) the sizing field, and evaluates this fine field on the coarse mesh.
    Either projects on the elements or nodes of the coarse mesh, depending on the sizing_field_interpolation_type.
    As there may be discrepancies between the coarse and fine mesh, the sizing field interpolation has several
    special cases to handle:
    1. If a coarse element contains multiple fine elements, the sizing field is set to the minimum, mean or maximum
        sizing field of the fine elements, depending on the interpolation type.
    2. If a coarse element does not contain any fine elements, the sizing field is set to the sizing field of the
        nearest fine element.
    3. If a fine element does not contain any coarse elements, the sizing field is set to the sizing field of the
        nearest coarse element.
    Args:
        queried_mesh: Coarse mesh to interpolate the sizing field to.
        fine_mesh: Expert mesh to interpolate the sizing field from.
        sizing_field_interpolation_type: How to interpolate a sizing field from the fine mesh to our mesh
            "interpolated_vertex": build a piecewise-linear function over the vertices of the coarse mesh by projecting from
             the fine vertices to the coarse vertices.
             We can evaluate this function on the vertices of the coarse mesh to obtain a sizing field.
            "sampled_vertex": The same as interpolated_vertex, but use the point evaluation of the fine mesh as the sizing
                field. This is slightly more accurate, but less smooth than the "interpolated_vertex" version.
            "fine_vertex": Instead, keep the fine vertices as the sizing field. Since the coarse field is predicted
             on the coarse mesh, the coarse solution is projected to the fine mesh, leading to an overspecified
             linear system of equations.
             This always makes an error, but minimizing this error with, e.g., an MSE, accounts for something akin to
             a least-squares regression of the fine field on the coarse mesh.

        Alternatively, we can set the sizing field as a piecewise-constant function over the elements of the coarse mesh
            (by calling interpolate_element_sizing_field()).

    Returns: A sizing field interpolated from the fine mesh.
        The field is either on the coarse or the fine mesh, and on the elements, vertices or quadrature points
        of the mesh.

    """
    assert sizing_field_interpolation_type in get_args(SizingFieldInterpolationType), f"{sizing_field_interpolation_type=} not recognized"

    # handle sizing fields on mesh vertices
    if "vertex" in sizing_field_interpolation_type:
        return interpolate_vertex_sizing_field(
            queried_mesh=queried_mesh,
            reference_mesh=fine_mesh,
            sizing_field_interpolation_type=sizing_field_interpolation_type,
        )
    elif sizing_field_interpolation_type == "element":  # handle sizing fields on mesh elements
        return interpolate_element_sizing_field(fine_mesh, queried_mesh)
    else:
        raise ValueError(f"Interpolation type {sizing_field_interpolation_type=} not recognized")


def interpolate_vertex_sizing_field(
    queried_mesh: MeshWrapper, reference_mesh: MeshWrapper, sizing_field_interpolation_type: SizingFieldInterpolationType
) -> SizingField:
    """
    Interpolates the sizing field from fine expert mesh to the coarse mesh. Uses the volumes of the fine mesh elements
    as (a basis for) the sizing field, and evaluates this fine field on the coarse mesh.
    Projects the sizing field on a vertex-level. Options include:
    * interpolated_vertex: build a piecewise-linear function over the vertices of the coarse mesh by projecting from
      the fine vertices to the coarse vertices.
      We can evaluate this function on the vertices of the coarse mesh to obtain a sizing field.
    * sampled_vertex: The same as interpolated_vertex, but use the point evaluation of the fine mesh as the sizing
        field. This is slightly more accurate, but less smooth than the "interpolated_vertex" version.
    * fine_vertex: Instead, keep the fine vertices as the sizing field. Since the coarse field is predicted
        on the coarse mesh, the coarse solution is projected to the fine mesh, leading to an overspecified
        linear system of equations.
    Args:
        queried_mesh: The mesh to interpolate the sizing field to.
        reference_mesh:  The (expert) reference mesh to interpolate the sizing field from.
        sizing_field_interpolation_type: How to interpolate a sizing field from the fine mesh to our mesh
            "interpolated_vertex": build a piecewise-linear function over the vertices of the coarse mesh by projecting from
             the fine vertices to the coarse vertices.
             We can evaluate this function on the vertices of the coarse mesh to obtain a sizing field.
            "sampled_vertex": The same as interpolated_vertex, but use the point evaluation of the fine mesh as the sizing
                field. This is slightly more accurate, but less smooth than the "interpolated_vertex" version.
            "fine_vertex": Instead, keep the fine vertices as the sizing field. Since the coarse field is predicted
             on the coarse mesh, the coarse solution is projected to the fine mesh, leading to an overspecified
             linear system of equations.
             This always makes an error, but minimizing this error with, e.g., an MSE, accounts for something akin to
             a least-squares regression of the fine field on the coarse mesh.

    Returns: A sizing field interpolated from the fine mesh.

    """
    if sizing_field_interpolation_type == "interpolated_vertex":
        fine_sizing_field = get_sizing_field(reference_mesh, mesh_node_type="vertex")
        coarse_sizing_field = interpolate_vertex_field(reference_mesh, queried_mesh, from_scalars=fine_sizing_field)
        return coarse_sizing_field
    elif sizing_field_interpolation_type == "sampled_vertex":
        fine_sizing_field = get_sizing_field(reference_mesh, mesh_node_type="element")
        coarse2fine_correspondences = reference_mesh.find_closest_elements(queried_mesh.vertex_positions)
        coarse_sizing_field = fine_sizing_field[coarse2fine_correspondences]
        return coarse_sizing_field
    else:
        raise ValueError(f"Sizing field interpolation type {sizing_field_interpolation_type=} not recognized")


def interpolate_element_sizing_field(fine_mesh, queried_mesh) -> SizingField:
    """
    Interpolates the sizing field from fine expert mesh to the intermediate queried mesh over the mesh elements.
    For each element in the queried mesh, look at all fine element midpoints that it contains, and numerically integrate
    the corresponding sizing fields.
    If a queried element contains no fine midpoint, set its sizing field to the fine element that it is contained by.
    If this also does not exist, resort to nearest-neighbor search to find an appropriate expert element.


    Args:
        fine_mesh: The mesh to interpolate the sizing field from.
        queried_mesh: The mesh to interpolate the sizing field to.


    Returns: A sizing field interpolated from the fine mesh. Has shape (queried_mesh.nelements,), and can be interpreted
        as a piecewise-constant function over the elements of the queried mesh.

    """
    import torch
    from torch_scatter import scatter_add

    from src.helpers.torch_util import detach

    fine_sizing_field = get_sizing_field(fine_mesh, mesh_node_type="element")
    fine2coarse_correspondences = get_midpoint_correspondences(from_mesh=fine_mesh, to_mesh=queried_mesh)
    fine2coarse_correspondences = torch.tensor(fine2coarse_correspondences, dtype=torch.int64)
    fine_sizing_field = torch.tensor(fine_sizing_field, dtype=torch.float32)

    volumes = torch.tensor(fine_mesh.simplex_volumes)
    summed_sizing_fields = scatter_add(src=fine_sizing_field * volumes, index=fine2coarse_correspondences, dim=0, dim_size=queried_mesh.nelements)
    element_weights = scatter_add(src=volumes, index=fine2coarse_correspondences, dim=0, dim_size=queried_mesh.nelements)
    missing_elements = summed_sizing_fields == 0
    element_weights[missing_elements] = 1  # avoid division by zero.
    # This is just a placeholder value, since both summed_sizing_fields and element_weights should be zero.
    coarse_sizing_field = summed_sizing_fields / element_weights

    coarse_sizing_field = detach(coarse_sizing_field)

    if np.any(coarse_sizing_field == 0):
        # if the coarse sizing field for any element is zero, it does not contain any fine elements
        # this can happen if the coarse mesh is locally finer than the fine mesh, or for concave geometries.
        # In this case, we set the sizing field to that of the fine element that contains the coarse element.
        # This is basically the "midpoint" interpolation, but only for the elements that are missing.
        # Note that this can still fail for some elements if the coarse mesh is globally finer than the fine mesh.
        # In this case, the if-case below captures the last elements via a kd-tree and minimum distance
        missing_elements = np.where(coarse_sizing_field == 0)[0]
        query_points = queried_mesh.element_midpoints

        coarse2fine_correspondences = fine_mesh.element_finder()(*query_points[missing_elements].T)

        found_elements = missing_elements[coarse2fine_correspondences != -1]
        found_correspondences = coarse2fine_correspondences[coarse2fine_correspondences != -1]
        coarse_sizing_field[found_elements] = fine_sizing_field[found_correspondences]

    # If the coarse2fine mapping fails, a query point of the coarse mesh is outside the fine mesh.
    # This happens for e.g., concave surfaces. We now take the nearest element as evaluated by midpoint distance
    if np.any(coarse_sizing_field == 0):
        missing_elements = np.where(coarse_sizing_field == 0)[0]

        fine_tree = fine_mesh.midpoint_tree
        _, candidate_indices = fine_tree.query(query_points[missing_elements], k=1)
        candidate_indices = candidate_indices.astype(np.int64)
        coarse_sizing_field[missing_elements] = fine_sizing_field[candidate_indices]
    return coarse_sizing_field
