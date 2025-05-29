from dataclasses import dataclass
from functools import cached_property

import numpy as np
from omegaconf import DictConfig

from src.algorithm.dataloader.mesh_generation_data import MeshGenerationData
from src.mesh_util.transforms.mesh_to_image import MeshImage, mesh_to_image


@dataclass
class ImageAmberData(MeshGenerationData):
    image_resolution: int
    _observation = None

    def __post_init__(self):
        # Ensure some constraints for this baseline compared to the (more general) AMBER.
        super().__post_init__()
        assert self.node_type == "pixel", f"Node type {self.node_type} not supported for ImageAmberData"
        assert self.sizing_field_interpolation_type == "pixel", (
            f"Sizing field interpolation type {self.sizing_field_interpolation_type=} " f"not supported for ImageAmberData"
        )

    @property
    def boundary(self):
        return self.source_data.boundary

    @property
    def observation(self) -> MeshImage:
        if self._observation is None:
            mesh_image = mesh_to_image(
                wrapped_mesh=self.mesh,
                reference_mesh=self.expert_mesh,
                node_feature_names=self.node_feature_names,
                feature_provider=self.feature_provider,
                boundary=self.source_data.boundary,
                image_resolution=self.image_resolution,
            )
            self._observation = mesh_image
        return self._observation

    @cached_property
    def _labels(self) -> np.ndarray:
        """
        ImageAmber gets its own labels from the observation.
        """
        raise NotImplementedError("ImageAmberData does not support _labels property directly. Use observation instead.")
