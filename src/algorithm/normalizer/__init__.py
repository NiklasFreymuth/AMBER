from typing import Callable

import torch
from omegaconf import DictConfig
from torch_geometric.data import Data

from src.algorithm.normalizer.dummy_running_normalizer import DummyRunningNormalizer
from src.algorithm.normalizer.graph_running_normalizer import GraphRunningNormalizer
from src.algorithm.normalizer.running_normalizer import RunningNormalizer
from src.algorithm.prediction_transform.prediction_transform import PredictionTransform
from src.mesh_util.transforms.mesh_to_image import MeshImage


def get_normalizer(
    normalizer_config: DictConfig,
    example_input: Data | MeshImage,
    prediction_transform: PredictionTransform,
) -> RunningNormalizer:
    """
    Constructs and returns a normalizer for observations and predictions based on the given configuration.

    This function determines whether normalization is required and selects the appropriate normalizer
    based on the type of `example_input`. If normalization is not needed, it returns a dummy normalizer.

    Args:
        normalizer_config (DictConfig): Configuration specifying whether to normalize inputs and/or predictions.
        example_input (Data | MeshImage): An example input, used to infer the type of normalizer to create.
        prediction_transform (Callable[[torch.Tensor], torch.Tensor]):
            Used to (un-)transform predictions from network scale to mesh scale and back

    Returns:
        RunningNormalizer: A normalizer instance that can normalize and denormalize observations and predictions.

    Raises:
        ValueError: If `example_input` is neither `Data` nor `MeshImage`, an error is raised.
    """
    normalize_inputs = normalizer_config.get("normalize_inputs")
    normalize_predictions = normalizer_config.get("normalize_predictions")
    if normalize_inputs or normalize_predictions:
        # Create a normalizer
        input_clip = normalizer_config.get("input_clip")
        if isinstance(example_input, Data):
            # Graph normalizer
            return GraphRunningNormalizer(
                example_graph=example_input,
                normalize_inputs=normalize_inputs,
                normalize_predictions=normalize_predictions,
                prediction_transform=prediction_transform,
                input_clip=input_clip,
            )

        elif isinstance(example_input, MeshImage):
            # Use image-based normalizer
            from src.algorithm.normalizer.mesh_image_running_normalizer import (
                MeshImageRunningNormalizer,
            )

            return MeshImageRunningNormalizer(
                example_mesh_image=example_input,
                normalize_inputs=normalize_inputs,
                normalize_predictions=normalize_predictions,
                prediction_transform=prediction_transform,
                input_clip=input_clip,
            )
        else:
            raise ValueError(f"Unsupported input type {type(example_input)}")
    else:
        # No normalization required
        return DummyRunningNormalizer()
