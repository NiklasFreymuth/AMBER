import abc
from dataclasses import dataclass
from os import PathLike
from typing import Dict, Optional, Type, Union

import numpy as np
from skfem import MeshTri1 as MeshTri1

from src.helpers.qol import classproperty


@dataclass(repr=False)
class MeshExtensionMixin(abc.ABC):
    """
    Defines extension methods for scikit FEM meshes that interface with fast element finders and geometry functions.
    """

    def __init__(self, *args, **kwargs):
        self._geom_fn = None
        super().__init__(*args, **kwargs)

    @property
    def geom_fn(self):
        assert self._geom_fn is not None, "Geometry function not set."
        return self._geom_fn

    @geom_fn.setter
    def geom_fn(self, value):
        self._geom_fn = value

    @property
    @abc.abstractmethod
    def base_mesh_class(self):
        """Must be implemented in subclasses to return the base mesh class (e.g. MeshTet1 or MeshTri1)."""
        pass

    @classproperty
    @abc.abstractmethod
    def _dim(cls) -> int:
        """Dimension of the mesh (2 or 3)."""
        pass

    @classmethod
    def init_from_geom_fn(cls, geom_fn, max_element_volume: float) -> "MeshExtensionMixin":
        """
        Initialize a (roughly uniform) simplical  mesh from a geometric model provided via `geom_fn`.

        The function computes an appropriate element size from the given maximum element volume,
        and uses this to generate an initial mesh via Gmsh.

        Args:
            geom_fn (callable): A function that returns a pygmsh-compatible geometry object
                (e.g., from a STEP file or constructed programmatically).
            max_element_volume (float): Upper bound on the volume of individual mesh elements. Used to determine
                the (uniform) mesh resolution

        Returns:
            ExtendedMeshTet1: A tetrahedral mesh generated from the geometry with the specified resolution.

        Raises:
            AssertionError: If unexpected positional or keyword arguments are passed.
        """
        from src.tasks.domains.geometry_util import volume_to_edge_length
        from src.tasks.domains.gmsh_util import generate_initial_mesh

        desired_element_size = volume_to_edge_length(max_element_volume, dim=cls._dim)
        mesh: "MeshExtensionMixin" = generate_initial_mesh(geom_fn, desired_element_size, dim=cls._dim, target_class=cls)
        return mesh

    def convert_new_mesh(self, new_mesh: MeshTri1) -> "MeshExtensionMixin":
        """
        Converts a new mesh on the same underlying geometry to an ExtendedMeshTri1 instance,
        including the geometry function and potentially other fields
        Args:
            new_mesh:

        Returns:

        """
        mesh = self.__class__(new_mesh.p, new_mesh.t)  # Reset class to the original (potentially custom) class
        mesh.geom_fn = self.geom_fn  # Reset the geometry function
        return mesh

    def element_finder(self, mapping=None):
        """
        Returns a function that finds the element containing a point.
        Uses KDTree to reduce candidate checks.
        """

        from pykdtree.kdtree import KDTree

        from src.mesh_util.point_in_geometry import points_in_simplices

        tree = KDTree(np.mean(self.p[:, self.t], axis=1).T)
        nelems = self.t.shape[1]
        elements = self.p[:, self.t].T
        fibonacci = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

        def finder(*coords, try_brute_force: bool = False, _k: int = 1, _brute_force: bool = False):
            points = np.stack(coords, axis=1)

            if _brute_force:
                if try_brute_force:
                    element_indices = points_in_simplices(points=points, simplices=elements, candidate_indices=None)
                else:
                    element_indices = np.full(points.shape[0], -1, dtype=np.int64)
            else:
                num_candidates = min(fibonacci[_k], nelems)
                _, candidate_indices = tree.query(points, num_candidates)

                if _k > 1:
                    candidate_indices = candidate_indices[:, fibonacci[_k - 1] :]
                candidate_indices = candidate_indices.astype(np.int64)

                element_indices = points_in_simplices(points=points, simplices=elements, candidate_indices=candidate_indices)

                invalid = element_indices == -1
                if invalid.any():
                    if _k < len(fibonacci) - 1:
                        element_indices[invalid] = finder(*(coord[invalid] for coord in coords), _k=_k + 1, _brute_force=False)
                    else:
                        element_indices[invalid] = finder(*(coord[invalid] for coord in coords), _brute_force=True)

            return element_indices

        return finder

    @abc.abstractmethod
    def refined(self, times_or_ix: Union[int, np.ndarray] = 1):
        # Overwrite refinement to make sure the new "extension" class is used
        raise NotImplementedError

    def __post_init__(self):
        """Support node orders used in external formats.

        We expect ``self.doflocs`` to be ordered based on the
        degrees-of-freedom in :class:`skfem.assembly.Dofs`.  External formats
        for high order meshes commonly use a less strict ordering scheme and
        the extra nodes are described as additional rows in ``self.t``.  This
        method attempts to accommodate external formas by reordering
        ``self.doflocs`` and changing the indices in ``self.t``.

        """
        from skfem.element import Element

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
            self.doflocs = np.ascontiguousarray(self.doflocs)

        if not self.t.flags["C_CONTIGUOUS"]:
            self.t = np.ascontiguousarray(self.t)

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

        return to_file(self.base_mesh_class(self.p, self.t), filename, point_data, cell_data, **kwargs)
