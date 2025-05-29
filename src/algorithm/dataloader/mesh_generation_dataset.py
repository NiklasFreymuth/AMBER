from typing import Dict, List

from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.algorithm.dataloader.mesh_generation_data import MeshGenerationData
from src.tasks.domains.mesh_wrapper import MeshWrapper


class MeshGenerationDataset(Dataset):
    def __init__(self, *, algorithm_config: DictConfig, persistent_data: List[MeshGenerationData]):
        """
        Args:
            algorithm_config: The configuration for the algorithm.
            persistent_data: List of persistent graph data (optional).
        """
        self.algorithm_config = algorithm_config
        self._persistent_data = persistent_data if persistent_data is not None else []  # Persistent storage,

    @property
    def data(self) -> List[MeshGenerationData]:
        """
        Returns the combined list of protected data and those currently in the buffer.

        Returns: A list of AMBERData objects
        """
        return self._persistent_data

    @property
    def expert_meshes(self) -> List[MeshWrapper]:
        """
        Returns the list of expert meshes from the persistent data.

        Returns: A list of MeshWrapper objects
        """
        return [data.expert_mesh for data in self._persistent_data]

    def __getitem__(self, idx):
        """Retrieve a single graph."""
        return self.data[idx]

    def __len__(self):
        """Total number of available graphs."""
        return len(self.data)

    @property
    def first(self) -> MeshGenerationData:
        return self.data[0]

    def get_metrics(self) -> Dict[str, float]:
        return {"dataset_size": len(self)}
