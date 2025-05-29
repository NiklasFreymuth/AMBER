from typing import List, Tuple, Union

import torch

from src.mesh_util.transforms.mesh_to_image import MeshImage


def get_feature_batch(batch: List[MeshImage], device: Union[str, torch.device]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a bunch of mesh image input features into a batch of uniform shape. Each image is of shape
    (1, feature_dim, x_i, y_i[, z_i]), where x_i, y_i and z_i are the dimensions of the current image.
    Args:
        batch: a list of MeshImage objects
        device:

    Returns:
        - A batch of shape (batch_size, feature_dim, max_x, max_y [, max_z])
        - Dimensions tensor of shape (batch_size, num_dimensions)
    """
    dimensions = torch.tensor([mesh_image.features.shape[2:] for mesh_image in batch], device=device)
    batch_size = len(batch)
    num_features = batch[0].features.shape[1]
    max_dimensions = dimensions.max(dim=0)[0]
    batched_features = torch.zeros((batch_size, num_features, *max_dimensions), device=device)
    for idx, mesh_image in enumerate(batch):
        mesh_image: MeshImage
        features = mesh_image.features.to(device)
        slices = [slice(None)] + [slice(0, s) for s in features.shape[2:]]
        batched_features[idx][slices] = features

    return batched_features, dimensions
