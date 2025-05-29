import numpy as np
import torch
from torch.distributions import (
    Categorical,
    Independent,
    MixtureSameFamily,
    MultivariateNormal,
)


def build_gmm(
    weights: np.array,
    means: np.array,
    diagonal_covariances: np.array,
    rotation_angles: np.array,
) -> MixtureSameFamily:
    """
    Builds a 2d Gaussian Mixture Model from input weights, means, covariance diagonals and a rotation of the covariance
    Args:
        weights:
        means:
        diagonal_covariances: Diagonal/uncorrelated covariance matrices.
         Has shape (num_components, 3) and will be broadcast to an array of matrices
        rotation_angles: Angle to rotate the covariance matrix by. A rotation of 2pi results in the original matrix

    Returns:

    """
    num_components, dimension = diagonal_covariances.shape
    diagonal_covariances = np.eye(dimension)[None, ...] * diagonal_covariances[:, None, :]  # broadcast to matrix of shape (num_components, dim, dim)

    # Initialize rotation matrices using np.eye()
    rotation_matrices = np.tile(np.eye(dimension), (num_components, 1, 1))

    # Populate the rotation matrices for x and y axes
    rotation_matrices[:, 0, 0] = np.cos(rotation_angles)
    rotation_matrices[:, 0, 1] = -np.sin(rotation_angles)
    rotation_matrices[:, 1, 0] = np.sin(rotation_angles)
    rotation_matrices[:, 1, 1] = np.cos(rotation_angles)

    rotated_covariances = np.einsum(
        "ijk, ikl, iml -> ijm",
        rotation_matrices,
        diagonal_covariances,
        rotation_matrices,
    )
    # generalization/vectorization of "ij, jk, lk -> il", i.e.,
    # rotated_covariance = rotation_matrix @ diagonal_covariance @ rotation_matrix.T

    weights = torch.tensor(weights)
    means = torch.tensor(means)
    rotated_covariances = torch.tensor(rotated_covariances)

    mix = Categorical(weights, validate_args=False)
    comp = Independent(
        base_distribution=MultivariateNormal(loc=means, covariance_matrix=rotated_covariances),
        reinterpreted_batch_ndims=0,
        validate_args=False,
    )
    gmm = MixtureSameFamily(mix, comp, validate_args=False)
    return gmm
