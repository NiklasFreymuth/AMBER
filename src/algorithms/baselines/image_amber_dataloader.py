from typing import List, Optional, Dict, Any, Type

import numpy as np
from skfem import Mesh

from src.hmpn.common.hmpn_util import make_batch
from src.algorithms.amber.mesh_wrapper import MeshWrapper
from src.algorithms.amber.dataloader.abstract_dataloader import AbstractDataLoader
from src.algorithms.amber.dataloader.amber_data import MeshRefinementData
from src.algorithms.baselines.mesh_to_image import MeshImage
from src.environments.problems import AbstractFiniteElementProblem
from util.types import InputBatch


class ImageAMBERDataLoader(AbstractDataLoader):
    # Generate the data for the ImageAmber algorithm, which is AMBER but acting on pixel data/images
    def __init__(
            self,
            initial_meshes: List[Mesh],
            fem_problems: Optional[List[AbstractFiniteElementProblem]],
            expert_meshes: List[MeshWrapper],
            element_feature_names: List[str],
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
            element_feature_names: The names of the features for the mesh elements that will be mapped to the pixels
                of the constructed images
            supervised_config: A dictionary with the following keys:
                - max_buffer_size: The maximum size of the buffer. This does not include the protected data.
                - sizing_field_interpolation_type: How to interpolate the sizing field from the expert mesh. Can be
                    one of "midpoint", "mean", "max"
            device: The device to which the data should be moved
            random_state: A random state for reproducible batch selection
            gmsh_kwargs: Optional keyword arguments for the Gmsh mesh generation algorithm
        """
        self.element_feature_names = element_feature_names
        self.image_resolution = supervised_config.get("image_resolution")
        self._boundaries = [np.concatenate((mesh.p.min(axis=1), mesh.p.max(axis=1)), axis=0)
                            for mesh in expert_meshes]
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
                         fem_idx: int, device: Optional[str]) -> MeshImage:
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
        from src.algorithms.baselines.mesh_to_image import mesh_to_image
        expert_mesh = self.expert_meshes[fem_idx]
        boundary = self._boundaries[fem_idx]

        labels = interpolate_sizing_field(
            coarse_mesh=wrapped_mesh,
            fine_mesh=expert_mesh,
            sizing_field_query_scope=self.sizing_field_query_scope,
            interpolation_type=self.sizing_field_interpolation_type,
        )
        mesh_image = mesh_to_image(wrapped_mesh=wrapped_mesh,
                                   labels=labels,
                                   element_feature_names=self.element_feature_names,
                                   fem_problem=fem_problem,
                                   boundary=boundary,
                                   device=device,
                                   image_resolution=self.image_resolution)
        return mesh_image

    def get_batch(self, batch_size: int) -> List[MeshImage]:
        if len(self) > batch_size:
            indices = self._random_state.choice(len(self), batch_size, replace=False)
        else:
            indices = np.arange(len(self))
        batch = [self.data[i].observation for i in indices]
        return batch

    @property
    def data_class(self) -> Type[MeshRefinementData]:
        return MeshRefinementData
