from typing import Any, Dict, Union

import numpy as np

from src.environments.domains.extended_mesh_tet1 import ExtendedMeshTet1
from src.environments.domains.extended_mesh_tri1 import ExtendedMeshTri1


def get_initial_mesh(*,
                     mesh_idx: int,
                     domain_config: Dict[Union[str, int], Any],
                     dataset_config: Dict[Union[str, int], Any],
                     data_type_key: str,
                     max_initial_element_volume: float,
                     random_state: np.random.RandomState) -> Union[ExtendedMeshTri1, ExtendedMeshTet1]:
    """
    Generates an initial mesh based on the specified domain configuration.

    Args:
        mesh_idx: Index of the mesh to generate.
        domain_config: Configuration dictionary containing details about the domain.
        dataset_config: Configuration dictionary containing details about the dataset.
        data_type_key: Key of the data type to generate meshes for.
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
        initial_mesh = _get_lshaped_mesh(domain_config=domain_config,
                                         max_initial_element_volume=max_initial_element_volume,
                                         random_state=random_state)
    elif domain_type in ["step_file_domain", "step_file"]:
        initial_mesh = _get_step_file_mesh(mesh_idx=mesh_idx,
                                           dataset_config=dataset_config,
                                           data_type_key=data_type_key,
                                           max_initial_element_volume=max_initial_element_volume)
    else:
        raise ValueError(f"Unknown domain type '{domain_type}'")
    return initial_mesh


def _get_lshaped_mesh(*, domain_config: Dict[Union[str, int], Any],
                      max_initial_element_volume: float,
                      random_state: np.random.RandomState) -> ExtendedMeshTri1:
    """
    Creates an L-shaped mesh based on the provided configuration.

    Args:
        domain_config: Configuration dictionary with specific settings for the L-shaped mesh.
        random_state: A random state for generating stochastic elements in the mesh.

    Returns:
        An instance of Mesh specific to L-shaped domains.
    """
    from src.environments.domains.extended_mesh_tri1 import ExtendedMeshTri1
    mean_hole_position = np.array([0.5, 0.5])
    maximum_position_distortion = domain_config.get("maximum_position_distortion", 0.2)

    offset = random_state.uniform(low=-maximum_position_distortion, high=maximum_position_distortion, size=2)
    hole_position = mean_hole_position + np.clip(offset, -0.3, 0.45)

    return ExtendedMeshTri1.init_lshaped(
        max_element_volume=max_initial_element_volume,
        hole_position=hole_position,
    )


def _get_step_file_mesh(*, mesh_idx: int,
                        dataset_config: Dict,
                        data_type_key: str,
                        max_initial_element_volume: float) -> ExtendedMeshTet1:
    """
    Creates a mesh from a STEP file based on the provided configuration.

    Args:
        mesh_idx: Index of the mesh to generate.
        dataset_config: Configuration dictionary containing details about the dataset.
        data_type_key: Key of the data type to generate meshes for. Either "train", "val" or "test".
        max_initial_element_volume: Maximum volume of the initial elements in the mesh.

    Returns:
        An instance of Mesh derived from a STEP file geometry.
    """
    from src.environments.domains.extended_mesh_tet1 import ExtendedMeshTet1
    from util.keys import DATASET_ROOT_PATH
    import os
    folder_path = dataset_config.get("folder_path")
    step_file_path: str = os.path.join(DATASET_ROOT_PATH, folder_path, data_type_key, f"{mesh_idx + 1:03d}.step")

    return ExtendedMeshTet1.init_from_geom(
        step_file_path=step_file_path,
        max_element_volume=max_initial_element_volume,
    )
