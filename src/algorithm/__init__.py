from lightning import LightningModule
from omegaconf import DictConfig

from src.algorithm.dataloader.mesh_generation_dataset import MeshGenerationDataset


def create_algorithm(
    algorithm_config: DictConfig,
    train_dataset: MeshGenerationDataset,
    loading: bool = False,
    checkpoint_path: str | None = None,
) -> LightningModule:
    algorithm_name = algorithm_config.name.lower()

    if algorithm_name in ["amber", "graphmesh"]:
        from src.algorithm.core.amber import Amber
        from src.algorithm.dataloader.amber_dataset import AmberDataset

        assert isinstance(train_dataset, AmberDataset)
        algorithm_cls = Amber
    elif algorithm_name in ["image_amber", "image_baseline"]:
        from src.algorithm.core.image_amber import ImageAmber

        algorithm_cls = ImageAmber
    else:
        raise ValueError(f"Algorithm {algorithm_name} does not exist.")

    if loading:
        algorithm = algorithm_cls.load_from_checkpoint(checkpoint_path, train_dataset=train_dataset)
    else:
        algorithm = algorithm_cls(algorithm_config=algorithm_config, train_dataset=train_dataset)
    return algorithm
