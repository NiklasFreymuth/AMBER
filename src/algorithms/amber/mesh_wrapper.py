from functools import lru_cache

import numpy as np
from skfem import Mesh


class MeshWrapper:
    """
    Wrapper class for a mesh.
    """

    def __init__(self, mesh: Mesh):
        self._wrapped_mesh = mesh

    def get_simplex_volumes(self):
        from src.environments.util.mesh_util import get_simplex_volumes_from_indices
        return get_simplex_volumes_from_indices(positions=self._wrapped_mesh.p.T,
                                                simplex_indices=self._wrapped_mesh.t.T)

    def get_midpoints(self):
        from src.environments.util.mesh_util import get_element_midpoints
        return get_element_midpoints(self._wrapped_mesh)

    def get_grid_correspondences(self, resolution: int) -> np.ndarray:
        """
        Get the grid correspondences for the mesh, i.e., the assignment of each point on a regular grid to an element
        in the mesh. Will be -1 if the point is not in the mesh.
        Args:
            resolution: The resolution of the grid

        Returns:

        """
        linspace = np.linspace(self._wrapped_mesh.p.min(axis=1), self._wrapped_mesh.p.max(axis=1), resolution)
        candidate_points = np.array(np.meshgrid(*np.repeat([linspace], self._wrapped_mesh.dim(), axis=0)))
        candidate_points = candidate_points.T.reshape(-1, self._wrapped_mesh.dim())
        return self._wrapped_mesh.element_finder()(*candidate_points.T)

    def get_midpoint_tree(self):
        from pykdtree.kdtree import KDTree
        return KDTree(self.get_midpoints())

    def get_vertex_tree(self):
        from pykdtree.kdtree import KDTree
        return KDTree(self._wrapped_mesh.p.T)

    @property
    def mesh(self):
        return self._wrapped_mesh

    def unwrap(self):
        return self._wrapped_mesh

    def __getattr__(self, name):
        # Delegate attribute access to the wrapped object
        return getattr(self._wrapped_mesh, name)

    ##############
    # properties #
    ##############

    @property
    def num_elements(self) -> int:
        return self.mesh.t.shape[1]

    @property
    @lru_cache(maxsize=1)
    def element_midpoints(self) -> np.array:
        """
        Returns the midpoints of all elements.
        Returns: np.array of shape (num_elements, 2)

        """
        from src.environments.util.mesh_util import get_element_midpoints

        return get_element_midpoints(self.mesh)

    @property
    def element_indices(self) -> np.array:
        return self.mesh.t.T

    @property
    def vertex_positions(self) -> np.array:
        """
        Returns the positions of all vertices/nodes of the mesh.
        Returns: np.array of shape (num_vertices, 2)

        """
        return self.mesh.p.T

    @property
    def mesh_edges(self) -> np.array:
        """
        Returns: the edges of all vertices/nodes of the mesh. Shape (2, num_edges)
        """
        return self.mesh.facets

    @property
    @lru_cache(maxsize=1)
    def element_neighbors(self) -> np.array:
        """
        Find neighbors of each element. Shape (2, num_neighbors)
        Returns:

        """
        # f2t are element/face neighborhoods, which are set to -1 for boundaries
        return self.mesh.f2t[:, self.mesh.f2t[1] != -1]

    @property
    @lru_cache(maxsize=1)
    def element_volumes(self) -> np.array:
        from src.environments.util.mesh_util import get_simplex_volumes_from_indices
        return get_simplex_volumes_from_indices(positions=self.vertex_positions,
                                                simplex_indices=self.element_indices)

    def __repr__(self):
        return f"CachedMeshWrapper({self._wrapped_mesh})"
