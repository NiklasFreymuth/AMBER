from typing import List

import numpy as np

from src.algorithm.dataloader.mesh_generation_iterable_dataset import (
    MeshGenerationIterableDataset,
)
from src.mesh_util.transforms.mesh_to_image import MeshImage


class ImageAmberIterableDataset(MeshGenerationIterableDataset):
    def _get_next_train_batch(self) -> List[MeshImage]:
        """
        Endlessly get the next training batch
        Returns:

        """
        if len(self.dataset) > self.batch_size:
            indices = np.random.choice(len(self.dataset), self.batch_size, replace=False)
        else:
            indices = np.arange(len(self.dataset))
        batch = [self[i].observation for i in indices]
        return batch
