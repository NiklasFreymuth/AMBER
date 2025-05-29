from typing import Iterator

from omegaconf import DictConfig
from torch.utils.data import IterableDataset
from torch_geometric.data import Batch

from src.algorithm.dataloader.mesh_generation_data import MeshGenerationData
from src.algorithm.dataloader.mesh_generation_dataset import MeshGenerationDataset


class MeshGenerationIterableDataset(IterableDataset):
    def __init__(self, algorithm_config: DictConfig, dataset: MeshGenerationDataset, is_train: bool):
        self.dataset = dataset
        self.is_train = is_train
        self.batch_size = algorithm_config.dataloader.batch_size

        steps_per_epoch = algorithm_config.dataloader.steps_per_epoch
        accumulate_grad_batches = algorithm_config.dataloader.accumulate_grad_batches
        self.batches_per_epoch = steps_per_epoch * accumulate_grad_batches

    def __iter__(self) -> Iterator[Batch | MeshGenerationData]:
        if self.is_train:
            for step in range(self.batches_per_epoch):
                yield self._get_next_train_batch()
        else:
            yield from iter(self.dataset)

    def __getitem__(self, item: int) -> MeshGenerationData:
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset) if not self.is_train else self.batches_per_epoch

    def _get_next_train_batch(self) -> Batch:
        """
        Endlessly get the next training batch
        Returns:

        """
        raise NotImplementedError
