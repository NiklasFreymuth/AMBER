from dataclasses import dataclass

import torch

from src.helpers.custom_types import MeshGenerationStatus
from src.tasks.domains.mesh_wrapper import MeshWrapper


@dataclass
class InferenceStepOutput:
    """
    Output for a single mesh generation inference step.
    This step takes an old mesh and some input to a neural network. The network then predicts a sizing field to
    generate a new mesh.
    * predictions: The predictions of the neural network. Shape depends on the input, which can be a graph or an image
    * output_mesh: The new mesh generated from the predictions
    * successful: Whether the inference step was successful
    """

    predictions: torch.Tensor
    output_mesh: MeshWrapper
    mesh_generation_status: MeshGenerationStatus

    @property
    def refinement_success(self) -> bool:
        return self.mesh_generation_status == "success"

    @property
    def refinement_failed(self) -> bool:
        return self.mesh_generation_status == "failed"

    @property
    def refinement_scaled(self) -> bool:
        return self.mesh_generation_status == "scaled"

    @property
    def refinement_okay(self) -> bool:
        return not self.refinement_failed
