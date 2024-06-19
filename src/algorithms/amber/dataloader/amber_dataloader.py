"""
Class maintaining a dataset for supervised learning for sizing field estimation
"""
from typing import List, Optional, Union, Dict, Any, Type

import numpy as np
import tqdm
from skfem import Mesh
from torch_geometric.data import Data, HeteroData

from src.hmpn.common.hmpn_util import make_batch
from src.algorithms.amber.mesh_wrapper import MeshWrapper
from src.algorithms.amber.dataloader.abstract_dataloader import AbstractDataLoader
from src.algorithms.amber.dataloader.amber_data import AMBERData
from src.environments.problems import AbstractFiniteElementProblem
from util.types import InputBatch


class AMBERDataLoader(AbstractDataLoader):
    """
    An online data loader that manages a buffer of graphs and allows dynamic addition of new graphs.
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
        """
        Initialize the AMBERDataLoader with a list of FEM problems and expert meshes. These are stored internally
        as a list, and can be accessed via the properties `fem_problems` and `expert_meshes`.
        The DataLoader can add an arbitrary number of new meshes and graphs via the `add` method, but all of these
        refer to the same list of FEM problems and expert meshes.
        Args:
            fem_problems: A list of FEM problems
            expert_meshes: A list of expert meshes
            element_feature_names: The names of the features for the elements of the graphs that are constructed
                from the fem problems and meshes on them
            edge_feature_names: The names of the features for the edges of the graphs that are constructed
                from the fem problems and meshes on them
            supervised_config: A dictionary with the following keys:
                - max_buffer_size: The maximum size of the buffer. This does not include the protected data.
                - sizing_field_interpolation_type: How to interpolate the sizing field from the expert mesh. Can be
                    one of "midpoint", "mean", "max"
            device: The device to which the data should be moved
            random_state: A random state for reproducible batch selection
            gmsh_kwargs: Optional keyword arguments for the Gmsh mesh generation algorithm
        """
        self.element_feature_names = element_feature_names
        self.edge_feature_names = edge_feature_names
        super().__init__(
            initial_meshes=initial_meshes,
            fem_problems=fem_problems,
            expert_meshes=expert_meshes,
            supervised_config=supervised_config,
            device=device,
            random_state=random_state,
            gmsh_kwargs=gmsh_kwargs,
        )

    def _get_observation(self, wrapped_mesh: MeshWrapper,
                         fem_problem: Optional[AbstractFiniteElementProblem],
                         fem_idx: int, device: Optional[str]) -> Union[Data, HeteroData]:
        """
        Constructs a torch geometric graph from a mesh and its corresponding FEM problem.
        Args:
            fem_idx: Idx of the FEM problem to which the mesh corresponds
            fem_problem: The FEM problem to which the mesh corresponds. May be None
            wrapped_mesh: The (cached) mesh to convert to a graph
            device: The device to which the graph should be moved

        Returns:

        """
        from src.algorithms.amber.amber_util import interpolate_sizing_field
        from src.algorithms.amber.mesh_to_graph import mesh_to_graph
        expert_mesh = self.expert_meshes[fem_idx]
        labels = interpolate_sizing_field(
            coarse_mesh=wrapped_mesh,
            fine_mesh=expert_mesh,
            sizing_field_query_scope=self.sizing_field_query_scope,
            interpolation_type=self.sizing_field_interpolation_type,
        )
        graph = mesh_to_graph(wrapped_mesh=wrapped_mesh,
                              labels=labels,
                              element_feature_names=self.element_feature_names,
                              edge_feature_names=self.edge_feature_names,
                              fem_problem=fem_problem,
                              device=device)
        return graph

    def get_batch(self, num_nodes: int) -> InputBatch:
        """
        Generates a batch of graphs randomly selected from the dataset.
        Since the graphs have different numbers of edges, the number of graphs in the batch is not fixed, but instead
        induced by the size of the graphs. This allows us to use batches of constant size (but different numbers of
        graphs) in the training loop.
        Args:
            num_nodes: The (approximate) number of edges to include in the batch.
            Will greedily sample graphs until the number of edges is reached.

        Returns:
            InputBatch: A batch of graph data ready for model input.

        """
        sorted_indices = np.argsort([data.sampled_count for data in self.data])
        total_nodes = 0

        batch = []
        for index in sorted_indices:
            new_nodes = self.data[index].graph.num_nodes
            if new_nodes + total_nodes > num_nodes:
                break
            total_nodes += new_nodes
            self.data[index].increment_sampled_count()
            batch.append(self.data[index].graph)

        return make_batch(batch)

    @property
    def data_class(self) -> Type[AMBERData]:
        return AMBERData
