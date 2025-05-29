from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

from src.helpers.torch_util import detach
from src.tasks.domains.mesh_wrapper import MeshWrapper
from src.tasks.features.feature_provider import FeatureProvider


@dataclass
class MeshImage:
    active_pixel_positions: np.ndarray  # positions that contain mesh elements
    feature_coordinates: np.ndarray  # positions of the pixels in the image
    features: Tensor  # features on the mesh elements discretized on a grid.
    ## Shape (1, num_features, {image_resolution}^dim)
    label_grid: Tensor  # grid-shaped labels per mesh element discretized on a grid
    labels: Tensor  # flattened labels per grid element
    is_mesh: np.array  # binary mask which pixels are part of the mesh and which are not
    pixel_volume: float  # volume of each pixel


def mesh_to_image(
    wrapped_mesh: MeshWrapper,
    reference_mesh: MeshWrapper,
    node_feature_names: List[str],
    feature_provider: FeatureProvider | None,
    boundary: np.ndarray,
    image_resolution: int,
) -> MeshImage:
    """
    Generates an observation image/voxel grid from a finite element problem and a sizing field. This image is used as
    input for the CNN-based supervised learning algorithm.
    Args:
        wrapped_mesh:
        reference_mesh: The reference mesh that is used to generate the labels by querying its sizing field
        node_feature_names:
        feature_provider:
        boundary: The boundary as a numpy array (x1, y1, {z1}, x2, y2, {z2}) that defines the bounding box of the expert
        mesh. The boundary is used to determine the positions of the pixels in the geometry image.
        image_resolution: The resolution of the image in the longest direction.

    Returns: A MeshImage object with the mesh positions, features, label grid, labels, and is_mesh array.
    Features has shape (1, num_features, image_resolution^dim)

    """
    from src.mesh_util.transforms.mesh_to_graph import get_mesh_element_features

    # Calculate image features from mesh elements. Yields features of shape (num_elements, num_features)
    element_features = get_mesh_element_features(
        wrapped_mesh,
        feature_provider=feature_provider,
        node_feature_names=node_feature_names,
    )

    # Step 1: Generate the full pixel grid and get is_mesh mask
    points, image_shape, pixel_volume = _generate_pixel_grid(bounding_box=boundary, resolution=image_resolution)
    correspondences = wrapped_mesh.element_finder()(*points)
    is_mesh = correspondences != -1  # shape: (*image_shape,), containing True/False for each pixel

    # Step 2: Calculate features on the active pixels
    mesh_pixels = correspondences[is_mesh]  # indices of the mesh elements that correspond to some pixel(s). Shape (#active_pixels,)

    num_input_features = element_features.shape[1] + 1  # all prior features + 1 for the mesh/non-mesh indicator
    feature_grid = np.zeros((num_input_features, np.prod(image_shape)))
    feature_grid[:-1, is_mesh] = element_features[mesh_pixels].T
    feature_grid[-1, :] = (2 * is_mesh) - 1  # either "1" or "-1" for mesh or non-mesh points
    feature_grid = feature_grid.reshape(1, num_input_features, *image_shape)

    # Step 3: Compute labels only for mesh-covered pixels
    active_pixel_positions = points[:, is_mesh]
    labels = get_labels(
        reference_mesh, active_pixel_positions=active_pixel_positions, boundary=boundary, image_shape=image_shape, is_mesh_mask=is_mesh
    )  # shape: (num_active_pixels,)

    # Step 4: Create full label grid, initialize background (e.g., with NaN or -1)
    label_grid = np.full(np.prod(image_shape), fill_value=np.nan, dtype=np.float32)  # or 0, -1, etc.
    label_grid[is_mesh] = labels
    label_grid = label_grid.reshape(1, *image_shape)

    return MeshImage(
        active_pixel_positions=active_pixel_positions,
        feature_coordinates=points,
        features=torch.tensor(feature_grid, dtype=torch.float32),
        label_grid=torch.tensor(label_grid, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.float32),
        is_mesh=is_mesh,
        pixel_volume=pixel_volume,
    )


def _generate_pixel_grid(bounding_box: np.ndarray, resolution: int) -> Tuple[np.ndarray, Tuple, float]:
    """
    Generate a pixel grid for 2D or 3D, given the boundary and resolution.

    Parameters:
        bounding_box: tuple or list of 4 (2D) or 6 (3D) floats
        resolution: int (maximum number of pixels per dimension)

    Returns:
        points: (dim, N) array of pixel centers
        image_shape: tuple of ints (grid shape)
        pixel_volume: float
    """

    # 1: Access config parameters, determine bounding box
    d = len(bounding_box) // 2  # Dimensionality
    lower = np.array(bounding_box[:d])
    upper = np.array(bounding_box[d:])
    size = np.abs(upper - lower)

    # 2: Compute resolution per dimension
    # Use fixed spacing across all dimensions based on the largest extent
    pixel_length = np.max(np.abs(size)) / (resolution - 1)
    res_per_dim = (np.floor(np.abs(size) / pixel_length)).astype(int) + 1

    # 3: Adjust upper bounds to align with uniform spacing
    upper = lower + (res_per_dim - 1) * pixel_length

    # 4: Generate coordinate axes
    axes = [np.linspace(lower[i], upper[i], res_per_dim[i]) for i in range(d)]

    # 5: Generate meshgrid and flatten to point list
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.vstack([m.ravel() for m in mesh])

    # 6: Compute voxel/pixel volume and shape
    voxel_volume = np.prod([axes[i][1] - axes[i][0] for i in range(d)])
    shape = tuple(len(ax) for ax in axes)

    # 7: Return result
    return points, shape, voxel_volume


def get_labels(reference_mesh: MeshWrapper, active_pixel_positions, boundary, image_shape, is_mesh_mask: np.ndarray) -> np.ndarray:
    """
    Returns: An array of shape (num_active_pixels,) containing a label for each pixel in the image
             that lies on the mesh/geometry. The label is the average size of the elements that
             are mapped to the pixel.
    """
    from torch_scatter import scatter_add

    from src.mesh_util.sizing_field_util import get_sizing_field

    reference_sizing_field = get_sizing_field(reference_mesh, mesh_node_type="element")
    reference_sizing_field = torch.tensor(reference_sizing_field, dtype=torch.float32)
    volumes = torch.tensor(reference_mesh.simplex_volumes)

    # id of the pixel on the grid. Need to map using is_mesh_mask by basically constructing an inverse mapping
    # Global pixel IDs for each element midpoint
    global_pixel_ids = query_pixel_ids(reference_mesh.element_midpoints, image_shape, boundary)  # shape (*image_shape,)
    valid_midpoint_mask = is_mesh_mask[global_pixel_ids]  # shape: (num_ref_elements). sum: num_midpoints_in_active_pixels,)

    # Mapping from global index to active index
    active_index_map = -np.ones(np.prod(image_shape), dtype=int)
    active_index_map[np.flatnonzero(is_mesh_mask)] = np.arange(active_pixel_positions.shape[1])

    # Mask to select only midpoints falling into mesh pixels
    valid = active_index_map[global_pixel_ids] != -1
    reference_pixel_ids = torch.tensor(active_index_map[global_pixel_ids[valid]], dtype=torch.long)
    # Ids of the pixels that the midpoint of each reference element maps into

    active_sizing_field = reference_sizing_field[valid_midpoint_mask]
    active_volume = volumes[valid_midpoint_mask]

    dim_size = active_pixel_positions.shape[1]
    summed_sizing_fields = scatter_add(src=active_sizing_field * active_volume, index=reference_pixel_ids, dim=0, dim_size=dim_size)
    element_weights = scatter_add(src=active_volume, index=reference_pixel_ids, dim=0, dim_size=dim_size)
    missing_elements = summed_sizing_fields == 0
    element_weights[missing_elements] = 1  # avoid division by zero.
    # This is just a placeholder value, since both summed_sizing_fields and element_weights should be zero.
    labels = summed_sizing_fields / element_weights
    labels = detach(labels)

    if np.any(detach(missing_elements)):
        # Pixels that do not have a label allocated to them, i.e., do not have an element midpoint within them.
        # Map these to the element in the mesh that matches the midpoint of the pixel.
        # This is, in this order of priority, the containing element, or the element with closest midpoint.

        missing_pixel_positions = active_pixel_positions[:, missing_elements]
        missing_correspondences = reference_mesh.find_closest_elements(query_points=missing_pixel_positions.T)
        labels[missing_elements] = detach(reference_sizing_field)[missing_correspondences]
    return labels


def query_pixel_ids(query_points, grid_shape, boundary):
    """
    Map query points to pixel IDs, assuming each pixel is centered in its cell.

    Parameters:
        query_points: (M, D) array
        grid_shape: tuple of ints (nx, ny[, nz])
        boundary: tuple/list of floats, length 4 (2D) or 6 (3D)

    Returns:
        pixel_ids: (M,) array of flattened pixel indices
    """
    dim = query_points.shape[1]

    if dim == 2:
        x0, y0, x1, y1 = boundary
        nx, ny = grid_shape
        dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
        origin = (x0 + dx / 2, y0 + dy / 2)  # shift to pixel center
        indices = np.round((query_points - origin) / (dx, dy)).astype(int)
        indices = np.clip(indices, [0, 0], [nx - 1, ny - 1])
        pixel_ids = indices[:, 0] * ny + indices[:, 1]

    elif dim == 3:
        x0, y0, z0, x1, y1, z1 = boundary
        nx, ny, nz = grid_shape
        dx, dy, dz = (x1 - x0) / nx, (y1 - y0) / ny, (z1 - z0) / nz
        origin = (x0 + dx / 2, y0 + dy / 2, z0 + dz / 2)
        indices = np.round((query_points - origin) / (dx, dy, dz)).astype(int)
        indices = np.clip(indices, [0, 0, 0], [nx - 1, ny - 1, nz - 1])
        pixel_ids = indices[:, 0] * ny * nz + indices[:, 1] * nz + indices[:, 2]

    else:
        raise ValueError("query_points must be 2D or 3D")

    return pixel_ids
