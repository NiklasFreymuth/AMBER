from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor

from src.algorithms.amber.mesh_wrapper import MeshWrapper
from src.environments.problems import AbstractFiniteElementProblem


@dataclass
class MeshImage:
    mesh_positions: np.ndarray  # positions that contain mesh elements
    feature_coordinates: np.ndarray  # positions of the pixels in the image
    features: Tensor  # features on the mesh elements discretized on a grid.
    ## Shape (1, num_features, {image_resolution}^dim)
    label_grid: Tensor  # grid-shaped labels per mesh element discretized on a grid
    labels: Tensor  # flattened labels per grid element
    is_mesh: np.array
    pixel_volume: float


def mesh_to_image(wrapped_mesh: MeshWrapper,
                  labels: np.ndarray,
                  element_feature_names: List[str],
                  fem_problem: Optional[AbstractFiniteElementProblem],
                  boundary: np.ndarray,
                  device: str,
                  image_resolution: int = 64, ) -> MeshImage:
    """
    Generates an observation image/voxel grid from a finite element problem and a sizing field. This image is used as
    input for the CNN-based supervised learning algorithm.
    Args:
        wrapped_mesh:
        labels:
        element_feature_names:
        fem_problem:
        boundary: The boundary as a numpy array (x1, y1, {z1}, x2, y2, {z2}) that defines the bounding box of the expert
        mesh. The boundary is used to determine the positions of the pixels in the geometry image.
        device: The accelerator device to use for the image data
        image_resolution: The resolution of the image. The image will have image_resolution pixels per dimension.

    Returns: A MeshImage object with the mesh positions, features, label grid, labels, and is_mesh array.
    Features has shape (1, num_features, image_resolution^dim)

    """
    dim = wrapped_mesh.dim()
    assert dim in [2, 3], f"Only 2D and 3D images are supported, but got dim={dim}"

    from src.algorithms.amber.mesh_to_graph import get_mesh_element_features
    element_features = get_mesh_element_features(wrapped_mesh,
                                                 fem_problem=fem_problem,
                                                 element_feature_names=element_feature_names)
    # shape (num_elements, num_features)

    # get correspondences between mesh element midpoints and image pixels
    if dim == 2:
        x = np.linspace(boundary[0], boundary[2], image_resolution)
        y = np.linspace(boundary[3], boundary[1], image_resolution)
        xx, yy = np.meshgrid(x, y)
        points = np.vstack([xx.ravel(), yy.ravel()])
        image_shape = (image_resolution, image_resolution)

        pixel_volume = np.abs((x[1] - x[0]) * (y[1] - y[0]))
    else:  # 3d
        x = np.linspace(boundary[0], boundary[3], image_resolution)
        y = np.linspace(boundary[4], boundary[1], image_resolution)
        z = np.linspace(boundary[2], boundary[5], image_resolution)
        xx, yy, zz = np.meshgrid(x, y, z)
        points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()])
        image_shape = (image_resolution, image_resolution, image_resolution)

        pixel_volume = np.abs((x[1] - x[0]) * (y[1] - y[0]) * (z[1] - z[0]))
    correspondences = wrapped_mesh.element_finder()(*points)
    is_mesh = correspondences != -1

    num_input_features = element_features.shape[1] + 1
    feature_grid = np.zeros((num_input_features, image_resolution ** dim))
    feature_grid[:-1, is_mesh] = element_features[correspondences[is_mesh]].T
    feature_grid[-1, :] = (2 * is_mesh) -1  # either "1" or "-1" for mesh or non-mesh points
    feature_grid = feature_grid.reshape(1, num_input_features, *image_shape)

    # generate labels from the expert mesh, project them onto the image.
    projected_labels = labels[correspondences[is_mesh]]
    mesh_positions = points[:, is_mesh]

    label_grid = np.zeros((image_resolution ** dim))
    label_grid[is_mesh] = projected_labels
    label_grid = label_grid.reshape(1, *image_shape)

    return MeshImage(
        mesh_positions=mesh_positions,
        feature_coordinates=points,
        features=torch.tensor(feature_grid, dtype=torch.float32, device=device),
        label_grid=torch.tensor(label_grid, dtype=torch.float32, device=device),
        labels=torch.tensor(projected_labels, dtype=torch.float32, device=device),
        is_mesh=is_mesh,
        pixel_volume=pixel_volume,
    )
