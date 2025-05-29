from functools import cached_property

import numpy as np

from src.tasks.domains.extended_mesh_tet1 import ExtendedMeshTet1
from src.tasks.domains.extended_mesh_tri1 import ExtendedMeshTri1


class MeshWrapper:
    """
    Wrapper class for a mesh that provides shorthands for common operations and caches some constants.
    """

    def __init__(self, mesh: ExtendedMeshTet1 | ExtendedMeshTri1):
        self._wrapped_mesh = mesh

    def find_closest_elements(self, query_points: np.ndarray) -> np.ndarray:
        """
        Map each query point to one element in this mesh.
        This is done by finding the element that each query point lies in.
        If a point does not lie in any element, it is mapped to the nearest element in terms of Euclidean distance of
        the point and the element midpoints.
        Args:
            query_points: A numpy array of shape (#query_points, dim) containing the query points.

        Returns: A numpy array of shape (#query_points, ) containing indices of range #self.elements that specify
            the elements of this mesh that contain each query point, or the closest elements if there is no
            containment due to a mismatch in geometry.
            The points must *not* lie in the found elements, which may cause issues for, e.g.,
            calculating the interpolation of a field on the elements.
        """
        corresponding_elements = self.element_finder()(*query_points.T)
        missing_elements = corresponding_elements == -1
        if missing_elements.any():
            # handle parts of the fine geometry that do not appear in the coarse one.
            # This may happen if the geometries are not perfectly aligned, e.g., for convex geometries.
            # In this case, we set the sizing field to the nearest coarse element (as evaluated by midpoint distance)
            # While this is not super accurate, it is a reasonable approximation that gets better for finer learned
            # meshes
            mesh2_tree = self.midpoint_tree
            _, candidate_indices = mesh2_tree.query(query_points[missing_elements], k=1)
            corresponding_elements[missing_elements] = candidate_indices.astype(np.int64)
        return corresponding_elements

    @cached_property
    def midpoint_tree(self):
        from pykdtree.kdtree import KDTree

        return KDTree(self.element_midpoints)

    @cached_property
    def vertex_tree(self):
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
    def num_vertices(self) -> int:
        return self.mesh.nvertices

    @cached_property
    def element_midpoints(self) -> np.array:
        """
        Returns the midpoints of all elements.
        Returns: np.array of shape (num_elements, 2)

        """
        from src.mesh_util.mesh_util import get_element_midpoints

        return get_element_midpoints(self.mesh)

    @cached_property
    def simplex_volumes(self):
        from src.tasks.domains.geometry_util import get_simplex_volumes_from_indices

        return get_simplex_volumes_from_indices(positions=self.vertex_positions, simplex_indices=self.element_indices)

    @property
    def element_indices(self) -> np.ndarray:
        return self.mesh.t.T

    @property
    def vertex_positions(self) -> np.ndarray:
        """
        Returns the positions of all vertices/nodes of the mesh.
        Returns: np.array of shape (num_vertices, 2)

        """
        return self.mesh.p.T

    @property
    def mesh_edges(self) -> np.ndarray:
        """
        Returns: the edges between all vertices/nodes of the mesh. Shape (2, num_edges)
        """
        if self.mesh.dim() == 2:
            return self.mesh.facets
        elif self.mesh.dim() == 3:
            return self.mesh.edges
        else:
            raise ValueError("Mesh dimension must be 2 or 3")

    @property
    def boundary_edges(self) -> np.ndarray:
        """
        Returns: the edges of the boundary of the mesh. Shape (2, num_boundary_edges)

        """
        if self.mesh.dim() == 2:
            return self.mesh.boundary_facets()
        elif self.mesh.dim() == 3:
            return self.mesh.boundary_edges()
        else:
            raise ValueError("Mesh dimension must be 2 or 3")

    @cached_property
    def element_neighbors(self) -> np.ndarray:
        """
        Find neighbors of each element. Shape (2, num_neighbors)
        Can be seen as undirected edges of the mesh elements, i.e., as tuples (element1, element2) for each edge.
        Returns:

        """
        # f2t are element/face neighborhoods, which are set to -1 for boundaries
        return self.mesh.f2t[:, self.mesh.f2t[1] != -1]

    @cached_property
    def boundary_vertex_normals(self) -> np.ndarray:
        """

        Returns: An array of shape (num_boundary_vertices, dim) with num_boundary_vertices = len(self.boundary_nodes())
        containing the normals of the boundary vertices. The normals have a norm of 1.

        """
        from src.mesh_util.mesh_util import compute_boundary_vertex_normals

        return compute_boundary_vertex_normals(mesh=self.mesh)

    @cached_property
    def boundary_edge_curvatures(self) -> np.ndarray:
        """

        Returns: An array of shape (num_boundary_edges, ) with the curvatures of the boundary edges. Curvatures are be
            negative for concave edges and positive for convex edges.
            The curvatures are computed using the boundary vertex normals. Curvatures are 0 for flat edges.

        """
        from src.mesh_util.mesh_util import compute_curvature

        return compute_curvature(mesh=self.mesh, boundary_normals=self.boundary_vertex_normals)

    def __repr__(self):
        return f"MeshWrapper({self._wrapped_mesh})"
