from dataclasses import dataclass
from typing import Type, Union

import numpy as np
from skfem import MeshTri1 as MeshTri1

from src.helpers.qol import classproperty
from src.tasks.domains.mesh_extension_mixin import MeshExtensionMixin


@dataclass(repr=False)
class ExtendedMeshTri1(MeshExtensionMixin, MeshTri1):
    """
    A wrapper/extension of the Scikit FEM MeshTri1 that allows for more flexible mesh initialization.
    This class allows for arbitrary sizes and centers of the initial meshes, and offers utility for different initial
    mesh types.

    """

    @property
    def base_mesh_class(self):
        return MeshTri1

    @classproperty
    def _dim(cls):
        return 2

    def __init__(self, *args, **kwargs):
        """ """
        self._boundary_polygon = None
        super().__init__(*args, **kwargs)

    @classmethod
    def init_polygon(
        cls: Type["ExtendedMeshTri1"],
        boundary_nodes: np.ndarray,
        max_element_volume: float = 0.01,
        *args,
        **kwargs,
    ) -> "ExtendedMeshTri1":
        r"""Initialize a polygonal 2d mesh
        ----------
        boundary_nodes
            is the position of the polygon
        max_element_volume
            The maximum volume of a mesh element size of the mesh. As the meshes are usually in [0,1]^2, the maximum
            number of elements is roughly bounded by 1/max_element_volume.
        """
        from src.tasks.domains.gmsh_geometries import polygon_geom

        assert not kwargs, f"No keyword arguments allowed, given '{kwargs}'"
        assert not args, f"No positional arguments allowed, given '{args}'"

        geom_fn = lambda: polygon_geom(boundary_nodes=boundary_nodes)
        mesh = cls.init_from_geom_fn(geom_fn, max_element_volume=max_element_volume)
        # Compute indices of boundary nodes in mesh vertices
        indices = []
        for bn in boundary_nodes:
            for idx, vertex in enumerate(mesh.p.T):
                if np.allclose(vertex, bn, atol=1e-8):
                    indices.append(idx)
                    break
        mesh._boundary_polygon = {"nodes": boundary_nodes, "indices": indices}
        return mesh

    @property
    def boundary_polygon(self):
        return self._boundary_polygon

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
        hp_x, hp_y = hole_position[0], hole_position[1]
        boundary_nodes = np.array(
            [[0.0, 1.0, 1.0, hp_x, hp_x, 0.0], [0.0, 0.0, hp_y, hp_y, 1.0, 1.0]],
            dtype=np.float64,
        ).T
        return cls.init_polygon(boundary_nodes=boundary_nodes, max_element_volume=max_element_volume)

    def refined(self, times_or_ix: Union[int, np.ndarray] = 1):
        refined_mesh = super(MeshTri1, self).refined(times_or_ix)
        return self.__class__(refined_mesh.p, refined_mesh.t)
