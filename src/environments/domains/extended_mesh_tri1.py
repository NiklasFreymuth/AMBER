from dataclasses import dataclass
from os import PathLike
from typing import Type, Union, Optional, Dict

import numpy as np
from skfem import MeshTri1 as MeshTri1


@dataclass(repr=False)
class ExtendedMeshTri1(MeshTri1):
    """
    A wrapper/extension of the Scikit FEM MeshTri1 that allows for more flexible mesh initialization.
    This class allows for arbitrary sizes and centers of the initial meshes, and offers utility for different initial
    mesh types.

    """

    def __init__(self, *args, **kwargs):
        """
        """
        self._geom_fn = None
        self._geom_bounding_box = None
        super().__init__(*args, **kwargs)

    @classmethod
    def init_polygon(
            cls: Type["ExtendedMeshTri1"],
            boundary_nodes: np.ndarray,
            max_element_volume: float = 0.01,
            *args,
            **kwargs,
    ) -> "ExtendedMeshTri1":
        r"""Initialize a mesh for the L-shaped domain.
        The mesh topology is as follows::
            *-------*       1
            | \     |
            |   \   |
            |     \ |
            *-------*-------*
            |     / | \     |
            |   /   |   \   |
            | /     |     \ |
            0-------*-------*
        Parameters
        ----------
        boundary_nodes
            is the position of the hole in the lshaped domain np.ndarray of shape (2,)
        max_element_volume
            The maximum volume of a mesh element size of the mesh. As the meshes are usually in [0,1]^2, the maximum
            number of elements is roughly bounded by 1/max_element_volume.
        """
        assert not kwargs, f"No keyword arguments allowed, given '{kwargs}'"
        assert not args, f"No positional arguments allowed, given '{args}'"

        from src.environments.domains.gmsh_util import generate_initial_mesh, polygon_geom
        from src.algorithms.amber.amber_util import volume_to_edge_length
        geom_fn = lambda: polygon_geom(boundary_nodes=boundary_nodes)
        desired_element_size = volume_to_edge_length(max_element_volume, dim=3)
        mesh: "ExtendedMeshTri1" = generate_initial_mesh(geom_fn, desired_element_size, dim=2, target_class=cls)
        return mesh

    @classmethod
    def init_lshaped(
            cls: Type["ExtendedMeshTri1"],
            max_element_volume: float = 0.01,
            hole_position=np.array([0.5, 0.5]),
            *args,
            **kwargs,
    ) -> "ExtendedMeshTri1":
        r"""Initialize a mesh for the L-shaped domain.
        The mesh topology is as follows::
            *-------*       1
            | \     |
            |   \   |
            |     \ |
            *-------*-------*
            |     / | \     |
            |   /   |   \   |
            | /     |     \ |
            0-------*-------*
        Parameters
        ----------
        hole_position
            is the position of the hole in the lshaped domain np.ndarray of shape (2,), range between 0, 1.
        max_element_volume
            The maximum volume of a mesh element size of the mesh. As the meshes are usually in [0,1]^2, the maximum
            number of elements is roughly bounded by 1/max_element_volume.
        """
        assert not kwargs, f"No keyword arguments allowed, given '{kwargs}'"
        assert not args, f"No positional arguments allowed, given '{args}'"
        hp_x, hp_y = hole_position[0], hole_position[1]

        from src.environments.domains.gmsh_util import generate_initial_mesh, polygon_geom
        from src.algorithms.amber.amber_util import volume_to_edge_length

        points = np.array([[0.0, 1.0, 1.0, hp_x, hp_x, 0.0],
                           [0.0, 0.0, hp_y, hp_y, 1.0, 1.0]],
                          dtype=np.float64, ).T
        geom_fn = lambda: polygon_geom(boundary_nodes=points)
        desired_element_size = volume_to_edge_length(max_element_volume, dim=3)
        mesh: "ExtendedMeshTri1" = generate_initial_mesh(geom_fn, desired_element_size, dim=2, target_class=cls)
        return mesh

    @property
    def geom_fn(self):
        assert self._geom_fn is not None, "Geometry function not set."
        return self._geom_fn

    @geom_fn.setter
    def geom_fn(self, value):
        self._geom_fn = value

    @property
    def geom_bounding_box(self):
        assert self._geom_bounding_box is not None, "Bounding box not set."
        return self._geom_bounding_box

    @geom_bounding_box.setter
    def geom_bounding_box(self, value):
        self._geom_bounding_box = value

    def element_finder(self, mapping=None):
        """
        Find the element that contains the point (x, y). Returns -1 if the point is in no element
        Args:
            mapping: A mapping from the global node indices to the local node indices. Currently not used

        Returns:

        """
        from pykdtree.kdtree import KDTree
        from src.environments.util.point_in_2d_geometry import parallel_points_in_triangles

        # maintain a kd-tree over element centers
        tree = KDTree(np.mean(self.p[:, self.t], axis=1).T)
        nelems = self.t.shape[1]
        elements = self.p[:, self.t].T

        fibonacci = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

        def finder(x, y, try_brute_force: bool = True, _k: int = 1, _brute_force: bool = False):
            """
            For each point in (x,y), find the element that contains this point.
            Returns -1 for points that are not in any element
            Args:
                x: Array of point x coordinates of shape (num_points, )
                y: Array of point y coordinates of shape (num_points, )
                try_brute_force: If True, resort to brute force if the KD Tree fails to find the element.
                _k: fib(_k) is the number of elements to consider for each point, where fib is the fibonacci sequence.
                   Only used if _brute_force is False.
                   Will do iterative passes over _k={1,2,3,4,5} before doing a brute force.
                _brute_force: If True, use the brute force variant. If False, use the KDTree to select candidates.
                Internal parameter that is used to switch between the two algorithms as a backup if the KDTree fails.

            Returns:

            """
            if _brute_force:
                if try_brute_force:
                    # brute force approach - check all elements
                    element_indices = parallel_points_in_triangles(points=np.array([x, y]).T,
                                                                   triangles=elements,
                                                                   candidate_indices=None
                                                                   )
                else:
                    element_indices = np.ones(x.shape[0], dtype=np.int64) * -1
            else:
                # find candidate elements
                num_elements = min(fibonacci[_k], nelems)

                # use the KDTree to find the elements with the closest center
                _, candidate_indices = tree.query(np.array([x, y]).T, num_elements)
                # usually (distance, index), but we only care about the indices

                if _k > 1:
                    # only use the last half as the previous half was considered in the previous iteration
                    candidate_indices = candidate_indices[:, fibonacci[_k - 1]:]
                # cast to int64 for compatibility reasons
                candidate_indices = candidate_indices.astype(np.int64)

                # try to find the right element for each point using the KDTree candidates
                element_indices = parallel_points_in_triangles(
                    points=np.array([x, y]).T,
                    triangles=elements,
                    candidate_indices=candidate_indices,
                )

                # fallback to brute force search for elements that were not found in the KDTree
                invalid_elements = element_indices == -1
                if invalid_elements.any():
                    if _k < len(fibonacci) - 1:
                        element_indices[invalid_elements] = finder(
                            x=x[invalid_elements],
                            y=y[invalid_elements],
                            _k=_k + 1,
                            _brute_force=False,
                        )
                    else:
                        element_indices[invalid_elements] = finder(
                            x=x[invalid_elements],
                            y=y[invalid_elements],
                            _brute_force=True,
                        )

            return element_indices

        return finder

    def refined(self, times_or_ix: Union[int, np.ndarray] = 1):
        refined_mesh = super().refined(times_or_ix)
        return self.__class__(refined_mesh.p, refined_mesh.t)

    def __post_init__(self):
        """Support node orders used in external formats.

        We expect ``self.doflocs`` to be ordered based on the
        degrees-of-freedom in :class:`skfem.assembly.Dofs`.  External formats
        for high order meshes commonly use a less strict ordering scheme and
        the extra nodes are described as additional rows in ``self.t``.  This
        method attempts to accommodate external formas by reordering
        ``self.doflocs`` and changing the indices in ``self.t``.

        """
        import logging

        from skfem.element import Element

        logger = logging.getLogger(__name__)
        if self.sort_t:
            self.t = np.sort(self.t, axis=0)

        self.doflocs = np.asarray(self.doflocs, dtype=np.float64, order="K")
        self.t = np.asarray(self.t, dtype=np.int64, order="K")

        M = self.elem.refdom.nnodes

        if self.nnodes > M and self.elem is not Element:
            # reorder DOFs to the expected format: vertex DOFs are first
            # note: not run if elem is not set
            p, t = self.doflocs, self.t
            t_nodes = t[:M]
            uniq, ix = np.unique(t_nodes, return_inverse=True)
            self.t = np.arange(len(uniq), dtype=np.int64)[ix].reshape(t_nodes.shape)
            doflocs = np.hstack(
                (
                    p[:, uniq],
                    np.zeros((p.shape[0], np.max(t) + 1 - len(uniq))),
                )
            )
            doflocs[:, self.dofs.element_dofs[M:].flatten("F")] = p[:, t[M:].flatten("F")]
            self.doflocs = doflocs

        # C_CONTIGUOUS is more performant in dimension-based slices
        if not self.doflocs.flags["C_CONTIGUOUS"]:
            if self.doflocs.shape[1] > 1e3:
                logger.warning("Transforming over 1000 vertices " "to C_CONTIGUOUS.")
            self.doflocs = np.ascontiguousarray(self.doflocs)

        if not self.t.flags["C_CONTIGUOUS"]:
            if self.t.shape[1] > 1e3:
                logger.warning("Transforming over 1000 elements " "to C_CONTIGUOUS.")
            self.t = np.ascontiguousarray(self.t)

        # run validation
        if self.validate and logger.getEffectiveLevel() <= logging.DEBUG:
            self.is_valid()

    def save(
            self,
            filename: Union[str, PathLike],
            point_data: Optional[Dict[str, np.ndarray]] = None,
            cell_data: Optional[Dict[str, np.ndarray]] = None,
            **kwargs,
    ) -> None:
        """Export the mesh and fields using meshio.

        Parameters
        ----------
        filename
            The output filename, with suffix determining format;
            e.g. .msh, .vtk, .xdmf
        point_data
            Data related to the vertices of the mesh.
        cell_data
            Data related to the elements of the mesh.

        """
        from skfem.io.meshio import to_file
        return to_file(MeshTri1(self.p, self.t), filename, point_data, cell_data, **kwargs)
