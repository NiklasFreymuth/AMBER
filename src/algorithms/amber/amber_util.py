from typing import Union, Optional

import numpy as np
import torch
from skfem import Basis, ElementTriP1, ElementTetP1

from src.algorithms.amber.mesh_wrapper import MeshWrapper


def interpolate_sizing_field(
        coarse_mesh: MeshWrapper,
        fine_mesh: MeshWrapper,
        sizing_field_query_scope: str = "elements",
        interpolation_type: str = "max",
) -> np.ndarray:
    """
    Interpolates the sizing field from the expert mesh to the coarse mesh. Uses the volumes of the expert mesh elements
    as a basis for the sizing field, and evaluates this fine field on the coarse mesh.
    As there may be discrepancies between the coarse and fine mesh, the sizing field interpolation has several
    special cases to handle:
    1. If a coarse element contains multiple fine elements, the sizing field is set to the minimum, mean or maximum
        sizing field of the fine elements, depending on the interpolation type.
    2. If a coarse element does not contain any fine elements, the sizing field is set to the sizing field of the
        nearest fine element.
    3. If a fine element does not contain any coarse elements, the sizing field is set to the sizing field of the
        nearest coarse element.
    Args:
        coarse_mesh: Coarse mesh to interpolate the sizing field to.
        fine_mesh: Expert mesh to interpolate the sizing field from.
        sizing_field_query_scope: Type of sizing field to interpolate. Can be either "elements" or "nodes".
            If "elements", the sizing field is interpolated as a piecewise constant function on the elements.
            If "nodes", the sizing field is interpolated as a piecewise linear function on the nodes.
        interpolation_type: We have different options to interpolate from the expert sizing field to the sizing field
            of the coarse mesh. Options in decreasing average sizing field are
            "min": For each coarse element, the sizing field equals the minimum of the expert sizing field evaluated at
                all elements whose midpoint is contained in the coarse element. This is the most aggressive
                sizing field estimate, and essentially guaranteed to over-refine everywhere. It also takes the
                smallest number of inference iterations (namely 2) to converge to the expert sizing field from any
                initial guess.
            "midpoint" [Default]: The sizing field of each element of the coarse mesh equals the
              expert sizing field evaluated at the midpoint of the coarse mesh's element
            "mean": For each coarse element, the sizing field equals the mean of the expert sizing field evaluated at
              all elements whose midpoint is contained in the coarse element. As a coarse element can contain more
              small than large elements, this mean will be biased towards small fields, essentially taking a geometric
              average.
            "max": For each coarse element, the sizing field equals the maximum of the expert sizing field evaluated at
                all elements whose midpoint is contained in the coarse element. This is the most conservative
                sizing field estimate, and essentially guaranteed to not over-refine anywhere. It also takes the largest
                number of inference iterations to converge to the expert sizing field.
            Note that anything other than

    Returns:

    """
    assert sizing_field_query_scope in ["elements", "nodes"]
    assert interpolation_type in ["min", "midpoint", "mean", "max"]

    fine_sizing_field = get_sizing_field(fine_mesh, sizing_field_query_scope)

    coarse_midpoints = coarse_mesh.get_midpoints()
    if interpolation_type in ["min", "mean", "max"]:
        fine2coarse_correspondences = _get_fine2coarse_correspondences(coarse_mesh, fine_mesh)

        import torch
        from util.torch_util.torch_util import detach
        fine_sizing_field = torch.tensor(fine_sizing_field, dtype=torch.float32)

        scatter_reduce = _get_scatter_reduce(interpolation_type)  # either min, mean or max
        coarse_sizing_field = scatter_reduce(src=fine_sizing_field,
                                             index=fine2coarse_correspondences,
                                             dim=0,
                                             dim_size=coarse_mesh.nelements,
                                             )
        coarse_sizing_field = detach(coarse_sizing_field)

        if np.any(coarse_sizing_field == 0):
            # if the coarse sizing field for any element is zero, the coarse element does not contain any fine elements
            # this can happen if the coarse mesh is locally finer than the fine mesh, or for concave geometries.
            # In this case, we set the sizing field to that of the fine element that contains the coarse element.
            # This is basically the "midpoint" interpolation, but only for the elements that are missing
            missing_elements = np.where(coarse_sizing_field == 0)[0]

            coarse2fine_correspondences = fine_mesh.element_finder()(*coarse_midpoints[missing_elements].T)

            found_elements = missing_elements[coarse2fine_correspondences != -1]
            found_correspondences = coarse2fine_correspondences[coarse2fine_correspondences != -1]
            coarse_sizing_field[found_elements] = fine_sizing_field[found_correspondences]
    else:  # interpolation_type == "midpoint":
        coarse2fine_correspondences = fine_mesh.element_finder()(*coarse_midpoints.T)
        coarse_sizing_field = fine_sizing_field[coarse2fine_correspondences]
        # set the sizing field to zero for elements that do not contain any fine elements. These are handled later
        coarse_sizing_field[coarse2fine_correspondences == -1] = 0

    if np.any(coarse_sizing_field == 0):
        # If the coarse2fine mapping fails after searching for midpoints, the coarse element is outside the fine mesh.
        # This happens form e.g., concave surfaces. Here, we take the nearest element as evaluated by midpoint distance
        missing_elements = np.where(coarse_sizing_field == 0)[0]

        fine_tree = fine_mesh.get_midpoint_tree()
        _, candidate_indices = fine_tree.query(coarse_midpoints[missing_elements], k=1)
        candidate_indices = candidate_indices.astype(np.int64)
        coarse_sizing_field[missing_elements] = fine_sizing_field[candidate_indices]
    return coarse_sizing_field


def _get_fine2coarse_correspondences(coarse_mesh: MeshWrapper, fine_mesh: MeshWrapper) -> torch.Tensor:
    """
    Create a mapping from the (midpoints of the) fine mesh to the coarse mesh.
    This is done by finding the coarse element that contains the midpoint of each fine element.
    As there may be differences in the mesh boundaries for the fine and coarse mesh, we additionally handle the case
    where a fine element does not have a corresponding coarse element. Here, we set the sizing field to the

    Args:
        coarse_mesh: A (coarse) CachedMeshWrapper object
        fine_mesh: A (fine) CachedMeshWrapper object

    Returns: A torch tensor shape (num_fine_elements, ) containing the indices of the coarse elements that contain
        the midpoints of the fine elements, or the closest coarse element if a fine element is not contained in any
        coarse element.
    """
    fine_midpoints = fine_mesh.get_midpoints()
    fine2coarse_correspondences = coarse_mesh.element_finder()(*fine_midpoints.T)
    if np.any(fine2coarse_correspondences == -1):
        # handle parts of the fine geometry that do not appear in the coarse one.
        # This may happen if the geometries are not perfectly aligned, e.g., for convex geometries.
        # In this case, we set the sizing field to the nearest coarse element (as evaluated by midpoint distance)
        # While this is not super accurate, it is a reasonable approximation that gets better for finer learned
        # meshes
        missing_elements = np.where(fine2coarse_correspondences == -1)[0]

        coarse_tree = coarse_mesh.get_midpoint_tree()
        _, candidate_indices = coarse_tree.query(fine_midpoints[missing_elements], k=1)
        candidate_indices = candidate_indices.astype(np.int64)
        fine2coarse_correspondences[missing_elements] = candidate_indices

    fine2coarse_correspondences = torch.tensor(fine2coarse_correspondences, dtype=torch.int64)
    return fine2coarse_correspondences


def _get_scatter_reduce(interpolation_type):
    if interpolation_type == "mean":
        from torch_scatter import scatter_mean

        scatter_reduce = scatter_mean

    elif interpolation_type == "max":
        from torch_scatter import scatter_max

        scatter_reduce = lambda *args, **kwargs: scatter_max(*args, **kwargs)[0]
    elif interpolation_type == "min":
        from torch_scatter import scatter_min

        scatter_reduce = lambda *args, **kwargs: scatter_min(*args, **kwargs)[0]
    else:
        raise ValueError(f"Unknown interpolation type '{interpolation_type}'")
    return scatter_reduce


def get_sizing_field(mesh: MeshWrapper, sizing_field_query_scope: str = "elements") -> np.ndarray:
    """
    Generates the sizing field for a given mesh. The sizing field is a field that represents the desired element size
    in terms of its average edge length.
    Args:
        mesh:
        sizing_field_query_scope:

    Returns:

    """
    element_volumes = mesh.get_simplex_volumes()
    element_edge_lengths = volume_to_edge_length(element_volumes, mesh.dim())
    if sizing_field_query_scope == "elements":
        expert_sizing_field = element_edge_lengths
    elif sizing_field_query_scope == "nodes":
        expert_sizing_field = project_to_nodes(mesh, element_edge_lengths)
    else:
        raise ValueError(f"Sizing field type {sizing_field_query_scope} not recognized")
    return expert_sizing_field


def project_to_nodes(mesh: MeshWrapper, element_values: np.ndarray) -> np.ndarray:
    """
    Projects the provided element-wise values to the nodes of the mesh. This is done by computing the weighted average
    of the element values for each node, where the weights are the volumes of the elements that contain the node.
    This corresponds to solving the FEM for linear elements with the given element values as the right-hand side.

    Args:
        mesh: A CachedMeshWrapper object containing a simplical 2d or 3d mesh
        element_values: Numpy array of shape (num_elements, ) containing the a value for each element


    Returns:

    """
    unwrapped_mesh = mesh.mesh
    print("@@@", unwrapped_mesh.p.shape)
    volumes = mesh.get_simplex_volumes()

    # Create arrays to store the sums and counts for the weighted average
    vertex_sums = np.zeros(mesh.nverts)
    vertex_weights = np.zeros(mesh.nverts)

    # Loop over each element
    for i, elem in enumerate(mesh.elements.T):
        # Distribute the element value to its vertices, weighted by the element volume
        for vertex in elem:
            vertex_sums[vertex] += element_values[i] * volumes[i]
            vertex_weights[vertex] += volumes[i]

    # Compute the weighted average
    values_per_vertex = vertex_sums / vertex_weights

    return values_per_vertex


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
        average_element_size = sizing_field ** 2 * np.sqrt(3) / 4
    elif dim == 3:
        average_element_size = sizing_field ** 3 * np.sqrt(2) / 12
    else:
        raise ValueError(f"Dimension {dim} not supported")
    return average_element_size


def to_scalar_basis(basis, linear: bool = False) -> Basis:
    """
    Returns: The scalar basis used for the error estimation via integration.

    """
    if linear:
        if basis.mesh.dim() == 2:
            element = ElementTriP1()
        elif basis.mesh.dim() == 3:
            element = ElementTetP1()
        else:
            raise ValueError(f"Unknown basis dimension {basis.mesh.dim()}")
    else:
        element = basis.elem
        while hasattr(element, "elem"):
            element = element.elem
    scalar_basis = basis.with_element(element)
    return scalar_basis


def calculate_loss(
        predictions: torch.Tensor,
        labels: torch.Tensor,
        loss_type: str,
) -> torch.Tensor:
    """
    Calculate the loss for a batch of predictions and labels.
    Args:
        predictions: The predictions of the model
        labels: The labels per element. In this case, the projected sizing field of the expert mesh
        loss_type: The type of loss to calculate. Can be "mse" or "log_mse"

    Returns: The loss

    """
    if loss_type == "log_mse":
        labels = torch.log(labels)
    differences = torch.abs(predictions - labels)
    element_loss = differences ** 2

    loss = torch.mean(element_loss)
    return loss
