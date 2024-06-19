import abc

from tqdm import tqdm

"""
Class maintaining a dataset for supervised learning for sizing field estimation
"""
from typing import List, Optional, Union, Dict, Any, Type

import numpy as np
from skfem import Mesh
from torch import Tensor
from torch_geometric.data import Data, HeteroData

from src.algorithms.amber.mesh_wrapper import MeshWrapper
from src.algorithms.amber.dataloader.amber_data import MeshRefinementData
from src.algorithms.amber.dataloader.dataloader_util import get_reconstructed_mesh
from src.environments.problems import AbstractFiniteElementProblem
from util.types import InputBatch


class AbstractDataLoader(abc.ABC):
    def __init__(
            self,
            initial_meshes: List[Mesh],
            fem_problems: Optional[List[AbstractFiniteElementProblem]],
            expert_meshes: List[MeshWrapper],
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
            supervised_config: A dictionary with the following keys:
                - max_buffer_size: The maximum size of the buffer. This does not include the protected data.
                - sizing_field_interpolation_type: How to interpolate the sizing field from the expert mesh. Can be
                    one of "midpoint", "mean", "max"
            device: The device to which the data should be moved
            random_state: A random state for reproducible batch selection
            gmsh_kwargs: Optional keyword arguments for the Gmsh mesh generation algorithm
        """
        self.device = device
        self._random_state = random_state

        self._fem_problems = fem_problems
        self._geometry_functions = [initial_mesh.geom_fn for initial_mesh in initial_meshes]
        self._geom_bounding_boxes = [initial_mesh.geom_bounding_box for initial_mesh in initial_meshes]
        self._initial_meshes = [MeshWrapper(initial_mesh) for initial_mesh in initial_meshes]

        for expert_mesh, geometry_function, geom_bounding_box in zip(expert_meshes,
                                                                     self._geometry_functions,
                                                                     self._geom_bounding_boxes):
            expert_mesh.mesh.geom_fn = geometry_function
            expert_mesh.mesh.geom_bounding_box = geom_bounding_box

        self._expert_meshes = expert_meshes

        self.gmsh_kwargs = gmsh_kwargs

        self.sizing_field_query_scope = "elements"
        self.sizing_field_interpolation_type = supervised_config.get("sizing_field_interpolation_type")
        self._protected_data = self._initialize_protected_data()  # Data that should not be removed from the buffer

        # buffer for reference meshes that act as "upper bounds" for the quality of the learned meshes
        self._reconstructed_meshes = [None] * len(fem_problems)

    def get_observation(
            self, fem_idx: int, mesh: Union[Mesh, MeshWrapper], device: Optional[str] = None
    ) -> Union[Data, HeteroData]:
        """
        Constructs a torch geometric graph from a mesh and its corresponding FEM problem.
        Args:
            fem_idx: Id of the FEM problem to which the mesh corresponds
            mesh: The mesh to convert to a graph
            device: The device to which the graph should be moved

        Returns:

        """

        if isinstance(mesh, Mesh):
            wrapped_mesh = MeshWrapper(mesh)
        else:
            wrapped_mesh = mesh

        if self.fem_problems is None:
            fem_problem = None
        else:
            fem_problem = self.fem_problems[fem_idx]

        if device is None:
            device = self.device

        graph = self._get_observation(wrapped_mesh, fem_problem, fem_idx, device)
        return graph

    def get_reconstructed_mesh(self, fem_idx: int) -> MeshWrapper:
        """
        Get the reconstructed mesh for a given FEM problem. If the mesh is not already calculated, it is calculated
        and stored in an internal buffer.
        Args:
            fem_idx: The index of the FEM problem for which to get the reconstructed mesh

        Returns: The reconstructed mesh

        """
        reconstructed_mesh = self._reconstructed_meshes[fem_idx]
        if reconstructed_mesh is None:
            expert_mesh = self.expert_meshes[fem_idx]
            reconstructed_mesh = get_reconstructed_mesh(expert_mesh, gmsh_kwargs=self.gmsh_kwargs)
            self._reconstructed_meshes[fem_idx] = reconstructed_mesh
        return reconstructed_mesh

    def _initialize_protected_data(self) -> List[MeshRefinementData]:
        """
        Initializes the protected data list. This is a list of data that should not be removed from the buffer.
        Here, this corresponds to coarse initial meshes and corresponding interpolated sizing fields and observaitons
        Returns:

        """
        protected_data = []
        for fem_idx, (wrapped_mesh, fem_problem) in tqdm(enumerate(zip(self._initial_meshes, self.fem_problems)),
                                                         desc="Generating Protected Data"):
            graph = self._get_observation(wrapped_mesh=wrapped_mesh, fem_problem=fem_problem,
                                          fem_idx=fem_idx, device=self.device)

            data_class: Type[MeshRefinementData] = self.data_class
            new_data = data_class(fem_idx=fem_idx, mesh=wrapped_mesh, observation=graph)
            protected_data.append(new_data)
        return protected_data

    @abc.abstractmethod
    def _get_observation(self, wrapped_mesh: MeshWrapper,
                         fem_problem: AbstractFiniteElementProblem,
                         fem_idx: int, device: Optional[str]) -> Union[Tensor, Data, HeteroData]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_batch(self, batch_size: int) -> InputBatch:
        raise NotImplementedError

    @property
    def data_class(self) -> Type[MeshRefinementData]:
        raise NotImplementedError

    ##############
    # Properties #
    ##############
    @property
    def observations(self) -> List[Union[Tensor, Data, HeteroData]]:
        return [data.observation for data in self.data]

    @property
    def meshes(self) -> List[MeshWrapper]:
        return [data.mesh for data in self.data]

    @property
    def first(self) -> MeshRefinementData:
        return self.data[0]

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def data(self) -> List[MeshRefinementData]:
        """
        Returns the combined list of protected data and those currently in the buffer.

        Returns: A list of AMBERData objects
        """
        return self._protected_data

    @property
    def fem_problems(self) -> Optional[List[AbstractFiniteElementProblem]]:
        return self._fem_problems

    @property
    def expert_meshes(self) -> List[MeshWrapper]:
        return self._expert_meshes

    @property
    def geometry_functions(self) -> List[callable]:
        return self._geometry_functions

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]

    def __iter__(self):
        # Return the iterator for the internal list
        return iter(self.data)

    def __repr__(self):
        return f"{self.__class__.__name__} containing {len(self)} data points"
