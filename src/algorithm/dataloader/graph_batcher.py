import abc
from typing import List

import numpy as np
from torch_geometric.data import Data

from src.algorithm import MeshGenerationDataset


class GraphBatcher:
    """
    Batch-based batcher.
    Sorts dataset by sampled_count, greedily adds least sampled data, until the batch is full.
    I.e., first considers all graphs that have been sampled the least amount of times, tries to add all of them to
    the batch, then moves to more-sampled graphs, iterating until the batch is as full as possible.
    """

    def __init__(self, dataset: MeshGenerationDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    def get_next_batch(self) -> List[Data]:
        next_batch = self._get_next_batch()
        assert len(next_batch) > 0, "Batch is empty, no data points were added."
        return next_batch

    def _get_next_batch(self) -> List[Data]:
        sorted_indices = np.argsort([data.sampled_count for data in self.dataset])
        total_size = 0
        batch = []

        for index in sorted_indices:
            data_point = self.dataset[index]
            graph_size = data_point.graph_size
            if graph_size + total_size > self.batch_size:
                continue
            else:
                # Add the graph to the batch as it still fits
                total_size = total_size + graph_size
                data_point.increment_sampled_count()
                batch.append(data_point.observation)
        return batch
