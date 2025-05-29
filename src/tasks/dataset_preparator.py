import abc
import warnings
from functools import cached_property
from typing import Any, Dict, Tuple, get_args

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from src.algorithm.dataloader.mesh_generation_data import MeshGenerationData
from src.algorithm.dataloader.mesh_generation_dataset import MeshGenerationDataset
from src.algorithm.dataloader.source_data import SourceData
from src.helpers.custom_types import DatasetMode
from src.helpers.qol import filter_included_fields
from src.tasks.domains.mesh_wrapper import MeshWrapper


class DatasetPreparator(abc.ABC):
    def __init__(
        self,
        algorithm_config: DictConfig,
        task_config: DictConfig,
    ):
        """

        Args:
            task_config:
        """
        self.algorithm_name = algorithm_config.name
        self.algorithm_config = algorithm_config
        self.task_config = task_config

        self._max_initial_element_volume = self.task_config.max_initial_element_volume

        self._data_init_kwargs = _get_data_init_kwargs(algorithm_config=self.algorithm_config, task_config=task_config)

        if self.algorithm_name in ["image_amber", "image_baseline"]:
            self.image_resolution = algorithm_config.image_resolution
        else:
            self.image_resolution = None

    def max_initial_element_volume(self, bounding_box: np.ndarray, dimension: int) -> float:
        """
        Compute the maximum initial element volume for meshing based on the geometry's bounding box
        and target resolution.

        When `self._max_initial_element_volume == "auto"`, the function derives an appropriate element volume
        from the spatial resolution and bounding box. The goal is to generate a mesh that sufficiently captures
        geometric detail, typically at image-pixel resolution.

        Args:
            bounding_box (np.ndarray): Array of shape (2, dim) with min and max coordinates of the geometry.
            dimension (int): Spatial dimension of the domain (2 or 3).

        Returns:
            float: Maximum volume for initial mesh elements.

        Notes:
            - If False, the total bounding volume is divided by `resolution ** dimension`.
            - A final `1 / dimension` scaling is applied to mildly refine the mesh beyond the pixel-based estimate.
        """

        if self._max_initial_element_volume == "auto":
            assert self.image_resolution is not None, "self.image_resolution must be set for 'auto' initial element volume"

            from math import sqrt

            from src.mesh_util.mesh_util import get_longest_side_from_bounding_box

            resolution = self.image_resolution  # resolution pixels in the **largest** dimension

            longest_side = get_longest_side_from_bounding_box(bounding_box=bounding_box)
            pixel_length = longest_side / resolution  # Step size per dimension

            # Compute element volume as (pixel_length)^dimension
            max_initial_element_volume = pixel_length**dimension

            # devide by sqrt(dim) to make elements a bit smaller, ensuring that (almost) all pixels have their own element
            max_initial_element_volume = max_initial_element_volume / sqrt(dimension)
            return max_initial_element_volume
        else:
            return self._max_initial_element_volume

    @cached_property
    def data_class(self):
        if self.algorithm_name == "amber":
            from src.algorithm.dataloader.amber_data import AmberData

            return AmberData
        elif self.algorithm_name == "graphmesh":
            from src.algorithm.dataloader.graphmesh_data import GraphMeshData

            return GraphMeshData
        elif self.algorithm_name in ["image_amber", "image_baseline"]:
            from src.algorithm.dataloader.image_amber_data import ImageAmberData

            return ImageAmberData
        else:
            raise NotImplementedError(f"Algorithm {self.algorithm_name} not implemented")

    @cached_property
    def dataset_class(self):
        if self.algorithm_name in ["amber", "graphmesh"]:
            from src.algorithm.dataloader.amber_dataset import AmberDataset

            return AmberDataset
        elif self.algorithm_name in ["image_amber", "image_baseline"]:
            return MeshGenerationDataset
        else:
            raise NotImplementedError(f"Algorithm {self.algorithm_name} not implemented")

    def __call__(self) -> Dict[str, MeshGenerationDataset]:
        loaders = {dataset_mode: self.get_dataset(dataset_mode=dataset_mode) for dataset_mode in ["train", "val", "test"]}
        return loaders

    def get_dataset(self, dataset_mode: str) -> MeshGenerationDataset:
        num_data_points = self.task_config.num_data_points.get(dataset_mode)
        assert dataset_mode in get_args(DatasetMode), f"{dataset_mode=} not recognized"
        if num_data_points == 0:
            warnings.warn("No data points requested for this dataset mode. Returning empty dataset.")
        data_points = [
            self.prepare_data_point(data_idx=data_idx, dataset_mode=dataset_mode)
            for data_idx in tqdm(range(num_data_points), desc=f"Preparing {dataset_mode} data points")
        ]

        return self.dataset_class(algorithm_config=self.algorithm_config, persistent_data=data_points)

    def prepare_data_point(self, data_idx: int, dataset_mode: DatasetMode) -> MeshGenerationData:
        """

        Args:
            data_idx:
            dataset_mode: Either "train", "val" or "test"

        Returns:

        """
        source_data, initial_mesh = self._prepare_source_and_mesh(data_idx=data_idx, dataset_mode=dataset_mode)
        amber_data: MeshGenerationData = self.data_class(mesh=initial_mesh, source_data=source_data, **self._data_init_kwargs)
        return amber_data

    def _prepare_source_and_mesh(self, data_idx: int, dataset_mode: str) -> Tuple[SourceData, MeshWrapper]:
        raise NotImplementedError


def _get_data_init_kwargs(*, algorithm_config: DictConfig, task_config: DictConfig) -> Dict[str, Any]:
    interpolation_type = task_config.sizing_field_interpolation_type
    from src.algorithm.util.parse_input_types import get_mesh_node_type

    node_type = get_mesh_node_type(interpolation_type=interpolation_type)
    node_feature_names = filter_included_fields(task_config.features[node_type])

    if "fem" in task_config and "solution_dimension" in task_config.fem:
        # add one feature name for each solution dimension, if applicable
        solution_dimension = task_config.fem.solution_dimension
        if solution_dimension > 1:
            node_feature_names_ = []
            for node_feature_name in node_feature_names:
                if "solution" in node_feature_name:
                    node_feature_names_.extend([f"{node_feature_name}_{i}" for i in range(solution_dimension)])
                else:
                    node_feature_names_.append(node_feature_name)
            node_feature_names = node_feature_names_

    input_kwargs = {
        "node_feature_names": node_feature_names,
        "node_type": node_type,  # "element" or "vertex"
        "sizing_field_interpolation_type": interpolation_type,
    }

    algorithm_name = algorithm_config.name
    if algorithm_name in ["amber", "graphmesh"]:
        # Act on graphs
        input_kwargs["edge_feature_names"] = filter_included_fields(task_config.features["edge"])
        input_kwargs["initial_mesh_handling"] = algorithm_config.initial_mesh_handling
        input_kwargs["add_self_edges"] = task_config.features["add_self_edges"]
        if algorithm_name == "graphmesh":
            # GraphMesh specific
            input_kwargs["boundary_graph_feature_names"] = filter_included_fields(algorithm_config.boundary_graph_features)
    elif algorithm_name in ["image_amber", "image_baseline"]:
        # Act on images
        input_kwargs["image_resolution"] = algorithm_config.image_resolution
    return input_kwargs
