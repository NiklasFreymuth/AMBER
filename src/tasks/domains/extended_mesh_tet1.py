from dataclasses import dataclass
from os import PathLike
from typing import Dict, Optional, Type, Union

import numpy as np
from skfem import Element, MeshTet1

from src.helpers.qol import classproperty
from src.tasks.domains.mesh_extension_mixin import MeshExtensionMixin


@dataclass(repr=False)
class ExtendedMeshTet1(MeshExtensionMixin, MeshTet1):
    """
    A wrapper/extension of the Scikit FEM MeshTri1 that allows for more flexible mesh initialization.
    This class allows for arbitrary sizes and centers of the initial meshes, and offers utility for different initial
    mesh types.

    """

    @property
    def base_mesh_class(self):
        return MeshTet1

    @classproperty
    def _dim(cls):
        return 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def refined(self, times_or_ix: Union[int, np.ndarray] = 1):
        """Return a refined ExtendedMeshTet1.

        Parameters
        ----------
        times_or_ix
            Either an integer giving the number of uniform refinements or an
            array of element indices for adaptive refinement.

        """
        m = self
        if isinstance(times_or_ix, int):
            for _ in range(times_or_ix):
                m = m._uniform()
        else:
            m = m._adaptive(times_or_ix)
        return m
