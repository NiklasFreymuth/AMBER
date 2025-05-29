from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig

from src.helpers.torch_util import detach
from src.tasks.features.fem.elliptic_problem.load_function.build_gmm import build_gmm


class GMMDensity:
    """
    Represents an abstract 2d bounded family of target functions to train on.
    These can be seen as target functions that the agents in the MeshRefinement task should approximate.
    """

    def __init__(
        self,
        *,
        target_function_config: DictConfig[Union[str, int], Any],
        bounding_box: np.array,
        random_state: np.random.RandomState,
        valid_point_function: Optional[Callable[[np.array], np.array]] = None,
        dimension: int = 2,
    ):
        """

        Args:
            target_function_config: Config containing information on the (family) of functions to build.
                Includes a parameter
                fixed_target: bool, which determines whether to represent a fixed_target function,
                or create/draw a random function
            bounding_box: A rectangular bounding box that the density must adhere to. Used for determining the mean of
                the gaussian used to draw the Gaussian Density
            random_state: Internally used random_state. Will be used to sample functions from pre-defined families,
                either once at the start if fixed_target, or for every reset() else.
        """
        self._distribution = None
        self.random_state = random_state
        self.target_function_config = target_function_config

        self.boundary = bounding_box
        self.fixed_target = target_function_config.get("fixed_target")
        self._density_mode = target_function_config.get("density_mode")

        self._dimension = dimension
        self._num_components = target_function_config.get("num_components")
        self.distance_threshold = target_function_config.get("distance_threshold", 0.0)  # where to place the component

        # how far to move from the (normalized) middle of the boundary, as a float [0, 0.5)
        mean_position_range = target_function_config.get("mean_position_range")

        # bounds of the diagonal covariance values, as list [lower, upper]
        lower_covariance_bound = target_function_config.get("lower_covariance_bound")
        upper_covariance_bound = target_function_config.get("upper_covariance_bound")

        assert 0 <= mean_position_range <= 0.5, f"mean_position_range must be in [0, 0.5], given '{mean_position_range}'"
        assert len(bounding_box) == 2 * dimension, f"Boundary must be of length {2 * dimension}, " f"given {len(bounding_box)}"
        assert 1.0e-6 < lower_covariance_bound <= upper_covariance_bound, (
            f"Need positive covariance and a lower "
            f"bound smaller than the upper bound, "
            f"given '{lower_covariance_bound}' "
            f"and '{upper_covariance_bound}'"
        )

        mean_position_range = np.array([0.5 - mean_position_range, 0.5 + mean_position_range])
        covariance_range = np.array([lower_covariance_bound, upper_covariance_bound])
        self._create_gmm(covariance_range, mean_position_range, valid_point_function)

    def _create_gmm(self, covariance_range, mean_position_range, valid_point_function):
        if self.fixed_target:
            weights = np.arange(self._num_components) + 1
            weights = weights / np.sum(weights)

            # Generate mean positions for one dimension
            # Tile the array to create the means for all dimensions
            mean_one_dimension = np.linspace(mean_position_range[0], mean_position_range[1], self._num_components)
            means = np.tile(mean_one_dimension, (self._dimension, 1)).T
            means = self._scale_to_boundary(means)

            diagonal_covariances = np.exp(
                np.linspace(
                    np.log(covariance_range[0]),
                    np.log(covariance_range[1]),
                    (self._num_components * self._dimension),
                )
            )
            diagonal_covariances = diagonal_covariances.reshape((self._num_components, self._dimension))
            rotation_angles = np.linspace(0, 1 * np.pi, self._num_components, endpoint=False)

        else:
            weights = 1 + np.exp(self.random_state.normal(size=self._num_components))
            weights = weights / np.sum(weights)  # make weights sum to 1

            # draw k random components with mean, softmax weighting and random orientation of the covariance
            means = self._sample_means(mean_position_range, valid_point_function)

            diagonal_covariances = np.exp(
                self.random_state.uniform(
                    low=np.log(covariance_range[0]),
                    high=np.log(covariance_range[1]),
                    size=(self._num_components, self._dimension),
                )
            )
            rotation_angles = self.random_state.random(self._num_components) * 2 * np.pi
        self._distribution = build_gmm(
            weights=weights,
            means=means,
            diagonal_covariances=diagonal_covariances,
            rotation_angles=rotation_angles,
        )

    def _sample_means(self, mean_position_range, valid_point_function):
        if valid_point_function is not None:
            found_means = []
            while len(found_means) < self._num_components:
                # do rejection sampling on domain until enough means are found
                candidate_means = self.random_state.uniform(
                    mean_position_range[0],
                    mean_position_range[1],
                    size=(self._num_components * 10, self._dimension),
                )
                candidate_means = self._scale_to_boundary(candidate_means)
                valid_means = valid_point_function(candidate_means, distance_threshold=self.distance_threshold)
                found_means.extend(valid_means)

            found_means = np.array(found_means)
            means = found_means[: self._num_components]
        else:
            means = self.random_state.uniform(
                low=mean_position_range[0],
                high=mean_position_range[1],
                size=(self._num_components, 2),
            )
            means = self._scale_to_boundary(means)
        return means

    def _scale_to_boundary(self, means: np.array):
        # normalize means wrt. boundary.
        # The boundary is given as (lower_x, lower_y, [lower_z], upper_x, upper_y, [upper_z])
        dimension = means.shape[1]

        # Extract lower and upper bounds based on dimension
        lower_bounds = self.boundary[:dimension]
        upper_bounds = self.boundary[dimension:]

        # Normalize means wrt. boundary
        means = means * (upper_bounds - lower_bounds) + lower_bounds
        return means

    def evaluate(self, samples: np.array, include_gradient: bool = False, density_mode: Optional[str] = None) -> np.array:
        """

        Args:
            samples: Array of shape (#samples, 2)
            include_gradient: Whether to include a gradient in the returns or not. If True, the output is an array
              of shape (#samples, 3), where the last dimension is for the function evaluation, and the grdient wrt.
              x and y. If False, the output is one evaluation per sample, i.e., of shape (#samples, )
            density_mode: Either "density" or "log_density". If None, takes self._density_mode.

        Returns:

        """
        if density_mode is None:
            density_mode = self._density_mode

        assert self._distribution is not None, "Need to specify a distribution before evaluating. " "Try calling reset() first."
        input_samples = torch.tensor(samples)
        if include_gradient:
            input_samples.requires_grad = True
        log_probability = self._distribution.log_prob(input_samples)
        if density_mode == "log_density":
            value = log_probability
        elif density_mode == "density":
            value = torch.exp(log_probability)
        else:
            raise ValueError(f"Unknown density_mode '{self._density_mode}'")

        if include_gradient:
            value.backward(torch.ones(len(input_samples)))
            gradients = detach(input_samples.grad)
            values = detach(value)
            return np.concatenate((values[:, None], gradients), axis=-1)
        else:
            return detach(value)
