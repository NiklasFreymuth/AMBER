from omegaconf import DictConfig
from torch_geometric.data import Batch

from src.algorithm.dataloader.amber_dataset import AmberDataset
from src.algorithm.dataloader.graph_batcher import GraphBatcher
from src.algorithm.dataloader.mesh_generation_iterable_dataset import (
    MeshGenerationIterableDataset,
)
from src.helpers.torch_util import make_batch


class AmberIterableDataset(MeshGenerationIterableDataset):
    def __init__(self, algorithm_config: DictConfig, dataset: AmberDataset, is_train: bool):
        super().__init__(algorithm_config, dataset=dataset, is_train=is_train)
        self.sizing_field_interpolation_type = algorithm_config.get("sizing_field_interpolation_type")
        self.batcher = GraphBatcher(dataset=dataset, batch_size=algorithm_config.dataloader.batch_size)

    def _get_next_train_batch(self) -> Batch:
        """
        Endlessly get the next training batch
        Returns:

        """
        batch = self.batcher.get_next_batch()
        return make_batch(batch)
