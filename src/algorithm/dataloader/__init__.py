from typing import Dict

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.algorithm.dataloader.mesh_generation_dataset import MeshGenerationDataset


def get_datasets(algorithm_config: DictConfig, task_config: DictConfig) -> Dict[str, MeshGenerationDataset]:
    """
    Wrapper function for generating expert data and creating train and validation data loaders
    Args:
    Returns:

    """
    task_name = task_config.get("name")
    if task_name in ["poisson", "laplace"]:
        from src.tasks.features.fem.elliptic_problem.elliptic_data_generator import (
            EllipticDataGenerator,
        )

        data_preparator_cls = EllipticDataGenerator
    elif any(x in task_name for x in ["console", "beam", "airfoil", "rod", "mold"]):
        from src.tasks.expert_geometry_dataset_preparator import (
            ExpertGeometryDatasetPreparator,
        )

        data_preparator_cls = ExpertGeometryDatasetPreparator
    else:
        raise ValueError(f"Unknown task {task_name}")
    data_preparator = data_preparator_cls(algorithm_config=algorithm_config, task_config=task_config)
    datasets = data_preparator()
    return datasets


def get_dataloader(algorithm_config: DictConfig, dataset: MeshGenerationDataset, is_train: bool) -> DataLoader:
    algorithm_name = algorithm_config.name
    if algorithm_name in ["amber", "graphmesh"]:
        from src.algorithm.dataloader.amber_iterable_dataset import (
            AmberDataset,
            AmberIterableDataset,
        )

        assert isinstance(dataset, AmberDataset)
        iterable_dataset = AmberIterableDataset(algorithm_config=algorithm_config, dataset=dataset, is_train=is_train)
    elif algorithm_name in ["image_amber", "image_baseline"]:
        from src.algorithm.dataloader.image_amber_iterable_dataset import (
            ImageAmberIterableDataset,
        )

        iterable_dataset = ImageAmberIterableDataset(algorithm_config=algorithm_config, dataset=dataset, is_train=is_train)
    else:
        raise NotImplementedError(f"Algorithm {algorithm_name} not implemented")

    # wrap the iterable dataset in a pytorch DataLoader for batching
    dataloader = DataLoader(iterable_dataset, batch_size=None, num_workers=0)

    return dataloader
