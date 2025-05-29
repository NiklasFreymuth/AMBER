from typing import Any, Union

import numpy as np
from omegaconf import DictConfig

from src.tasks.domains.extended_mesh_tet1 import ExtendedMeshTet1
from src.tasks.domains.extended_mesh_tri1 import ExtendedMeshTri1
from src.tasks.domains.gmsh_geometries import lattice_geom


def get_initial_mesh_from_domain_config(
    *,
    domain_config: DictConfig[Union[str, int], Any],
    max_initial_element_volume: float,
    random_state: np.random.RandomState,
) -> Union[ExtendedMeshTri1, ExtendedMeshTet1]:
    """
    Generates an initial mesh based on the specified domain configuration.

    Args:
        mesh_idx: Index of the mesh to generate.
        domain_config: Configuration dictionary containing details about the domain.
        dataset: Configuration dictionary containing details about the dataset.
        data_type: Key of the data type to generate meshes for.
        max_initial_element_volume: Maximum volume of the initial elements in the mesh.
        random_state: A random state for stochastic processes to ensure reproducibility.

    Returns:
        An instance of Mesh based on the specified domain configuration.

    Raises:
        ValueError: If the domain type specified is not recognized.
    """
    domain_type = domain_config.get("domain_type").lower()
    if domain_type in [
        "lshape",
        "lshaped",
        "l_shaped",
        "l_shape",
        "l-shaped",
        "l-shape",
    ]:
        initial_mesh = _get_lshaped_mesh(
            domain_config=domain_config,
            max_initial_element_volume=max_initial_element_volume,
            random_state=random_state,
        )
    elif domain_type == "lattice":
        initial_mesh = _get_lattice_mesh(
            domain_config=domain_config, max_initial_element_volume=max_initial_element_volume, random_state=random_state
        )
    else:
        raise ValueError(f"Unknown domain type '{domain_type}'")
    return initial_mesh


def _get_lshaped_mesh(
    *,
    domain_config: DictConfig[Union[str, int], Any],
    max_initial_element_volume: float,
    random_state: np.random.RandomState,
) -> ExtendedMeshTri1:
    """
    Creates an L-shaped mesh based on the provided configuration.

    Args:
        domain_config: Configuration dictionary with specific settings for the L-shaped mesh.
        random_state: A random state for generating stochastic elements in the mesh.

    Returns:
        An instance of Mesh specific to L-shaped domains.
    """
    mean_hole_position = np.array([0.5, 0.5])
    maximum_position_distortion = domain_config.get("maximum_position_distortion", 0.3)

    offset = random_state.uniform(low=-maximum_position_distortion, high=maximum_position_distortion, size=2)
    hole_position = mean_hole_position + np.clip(offset, -0.3, 0.45)

    return ExtendedMeshTri1.init_lshaped(
        max_element_volume=max_initial_element_volume,
        hole_position=hole_position,
    )


def _get_lattice_mesh(
    *,
    domain_config: DictConfig[Union[str, int], Any],
    max_initial_element_volume: float,
    random_state: np.random.RandomState,
) -> ExtendedMeshTri1:
    """
    Creates an L-shaped mesh based on the provided configuration.

    Args:
        domain_config: Configuration dictionary with specific settings for the L-shaped mesh.
        random_state: A random state for generating stochastic elements in the mesh.

    Returns:
        An instance of Mesh specific to L-shaped domains.
    """
    n_holes = random_state.randint(low=domain_config.min_holes, high=domain_config.max_holes)
    hole_size = random_state.uniform(low=domain_config.min_hole_size, high=domain_config.max_hole_size)
    domain_size = domain_config.get("size", 1)
    geom_fn = lambda: lattice_geom(n_holes=n_holes, hole_size=hole_size, domain_size=domain_size)
    return ExtendedMeshTri1.init_from_geom_fn(geom_fn=geom_fn, max_element_volume=max_initial_element_volume)
