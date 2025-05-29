from collections import deque
from typing import List, Optional, Union, Dict, Any

import numpy as np
from skfem import Mesh
from torch_geometric.data import Data, HeteroData

from src.algorithms.amber.mesh_wrapper import MeshWrapper
from src.algorithms.amber.dataloader.amber_data import AMBERData
from src.algorithms.amber.dataloader.amber_dataloader import AMBERDataLoader
from src.environments.problems import AbstractFiniteElementProblem


class AMBEROnlineDataLoader(AMBERDataLoader):
    """
    Extension of the AMBERDataLoader that is used for online data generation. This class is used to generate expert
    labels for new meshes that are generated during the training process.
    """

    def __init__(
            self,
            initial_meshes: List[Mesh],
            fem_problems: Optional[List[AbstractFiniteElementProblem]],
            expert_meshes: List[MeshWrapper],
            element_feature_names: List[str],
            edge_feature_names: List[str],
            supervised_config: Dict[str, Any],
            device: str,
            random_state: np.random.RandomState,
            gmsh_kwargs: Optional[dict] = None,
    ):

        self.max_buffer_size = supervised_config.get("max_buffer_size")
        self._buffer_data = deque(maxlen=self.max_buffer_size)
        super().__init__(
            initial_meshes=initial_meshes,
            fem_problems=fem_problems,
            expert_meshes=expert_meshes,
            element_feature_names=element_feature_names,
            edge_feature_names=edge_feature_names,
            supervised_config=supervised_config,
            device=device,
            random_state=random_state,
            gmsh_kwargs=gmsh_kwargs,
        )

    def add_data(
            self,
            fem_idx: int,
            mesh: Union[Mesh, MeshWrapper],
            refinement_depth: int,
            graph: Optional[Union[Data, HeteroData]] = None,
            protected: bool = False,
    ) -> int:
        """
        Adds new data to the buffer. If the data is not added as protected data and adding data exceeds the
        buffer's max size, the oldest data is removed.
        Args:
            fem_idx: The index of the FEM problem to which the mesh and graph correspond
            mesh: The mesh to add
            graph: The graph to add
            refinement_depth: The depth of the graph/mesh to add. Refers to the number of refinements from the initial mesh to
                the mesh that the graph represents. If graph is None, the graph is constructed from the mesh
            protected: If True, the data is added as protected data and will not be removed from the buffer

        Returns:
            The index of the added data
        """
        if graph is None:
            graph = self.get_observation(fem_idx, mesh, device=self.device)
        else:
            graph = graph.to(self.device)

        if isinstance(mesh, Mesh):
            # Wrap the mesh in a CachedMeshWrapper
            mesh = MeshWrapper(mesh)

        avg_sampled_count = int(np.mean([data.sampled_count for data in self.data]))
        new_data = AMBERData(fem_idx=fem_idx, mesh=mesh, observation=graph,
                             refinement_depth=refinement_depth, sampled_count=avg_sampled_count)
        if protected:
            self._protected_data.append(new_data)
            index = len(self._protected_data) - 1
        else:
            self._buffer_data.append(new_data)
            index = len(self) - 1
        return index

    def get_data_point(self, max_depth: Optional[int] = None, strategy: str = "random") -> AMBERData:
        """
        Get a random data point from the buffer. If a maximum depth is specified, only data points with a depth
        less than or equal to the maximum depth are considered.
        Args:
            max_depth: The maximum depth of the data point
            strategy: The strategy to use for sampling. Can be "random" or "stratified".
                "Stratified" sampling queries a target depth first, then randomly selects a data point with this depth.
                "Random" sampling selects a random data point from the buffer that matches the maximum depth.

        Returns: A random data point

        """

        if strategy == "stratified":
            assert max_depth is not None and max_depth >= 0, f"Invalid max_depth {max_depth} for stratified sampling"
            # Stratified sampling: Sample a target depth first, then randomly take a sample that matches this depth.
            # If there are no samples with this depth (yet), take a random sample.

            # Efficient extraction of depths and filtering using list comprehension and set for uniqueness
            valid_depths = set(p.refinement_depth for p in self.data if p.refinement_depth <= max_depth)
            if not valid_depths:
                raise ValueError(f"No valid data points with max_depth {max_depth}")
            target_depth = self._random_state.choice(list(valid_depths))

            # Collect indices of all data points matching the target depth in one pass, select one of them randomly
            selected_indices = [i for i, p in enumerate(self.data) if p.refinement_depth == target_depth]
            idx = self._random_state.choice(selected_indices)
        elif strategy == "random":
            # Random sampling: Sample a random valid data point by taking a random permutation of the data and
            # selecting the first valid data point. This point always exists because we have protected data with a
            # depth of 0.
            permutation = self._random_state.permutation(self.size)
            position = 0
            if max_depth is not None:
                assert max_depth >= 0
                while self.data[permutation[position]].refinement_depth > max_depth:
                    position += 1
            idx = permutation[position]

        else:
            raise ValueError(f"Unknown strategy {strategy}")
        return self.data[idx]

    @property
    def data(self) -> List[AMBERData]:
        """
        Returns the combined list of protected data and those currently in the buffer.

        Returns: A list of AMBERData objects
        """
        return self._protected_data + list(self._buffer_data)

    @property
    def is_full(self) -> bool:
        """
        Checks if the buffer has reached its maximum size.

        Returns:
            bool: True if the buffer is full, False otherwise.
        """
        return len(self._buffer_data) == self._buffer_data.maxlen
