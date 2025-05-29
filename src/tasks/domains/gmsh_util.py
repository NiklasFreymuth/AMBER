from pathlib import Path
from typing import Callable, Dict, Optional, Type

import gmsh
import numpy as np
import pygmsh
from skfem import Mesh
from skfem.io import from_meshio

from src.tasks.domains.gmsh_session import gmsh_session
from src.tasks.domains.mesh_extension_mixin import MeshExtensionMixin


def geom_fn_from_file(file_path: str) -> Callable[[], pygmsh.occ.Geometry]:
    """
    Load a step file and return a pygmsh geometry object representing the geometry of this file
    Args:
        file_path: (relative or absolute) path to the step/geo file

    Returns: A pygmsh geometry object representing the geometry of the step file

    """

    def _geom_fn() -> pygmsh.occ.Geometry:
        # Create a pygmsh geometry object
        geom = pygmsh.occ.Geometry()
        # Import the shapes from the file
        geom.import_shapes(file_path)
        return geom

    return _geom_fn


def generate_initial_mesh(
    geometry_function: callable,
    desired_element_size: float,
    target_class: Optional[Type[MeshExtensionMixin]] = None,
    gmsh_kwargs: Optional[Dict] = None,
    # normalize_by_bounding_box: bool = False,
    dim: int = 2,
    verbose: bool = False,
) -> MeshExtensionMixin:
    with gmsh_session(gmsh_kwargs, verbose=verbose):
        geom = geometry_function()
        gmsh.model.occ.synchronize()  # Synchronize the geometry
        geom.characteristic_length_max = desired_element_size
        geom.characteristic_length_min = 0.7 * desired_element_size
        m = geom.generate_mesh(dim=dim, verbose=verbose)
        # bounding_box = np.array(list(gmsh.model.occ.get_bounding_box(dim, 1)))

    mesh = from_meshio(m)  # Convert to scikit-fem mesh

    # if dim == 2:
    #     bounding_box = bounding_box[[0, 1, 3, 4]]
    # if normalize_by_bounding_box:
    #     mesh.p = (mesh.p - bounding_box[:2]) / (bounding_box[2:] - bounding_box[:2])
    if target_class is not None:
        mesh = target_class(mesh.p, mesh.t)

    mesh.geom_fn = geometry_function  # store the geometry for later use.
    return mesh


def get_bounding_box(*, geometry_fn: callable, verbose: bool = False) -> np.ndarray:
    """
    Get the bounding box of the geometry defined by the geometry function.
    Args:
        geometry_fn: Geometry function to evaluate
        verbose: Whether to print verbose output

    Returns: Bounding box of the geometry

    """

    with gmsh_session(verbose=verbose):
        geom = geometry_fn()
        gmsh.model.occ.synchronize()  # Synchronize the geometry
        dim = 2 if len(gmsh.model.getEntities(3)) == 0 else 3  # check dimension
        entities = gmsh.model.getEntities(dim)

        if not entities:
            raise RuntimeError(f"No entities of dimension {dim} found.")

        # Initialize bbox
        mins = np.full(3, np.inf)
        maxs = np.full(3, -np.inf)

        for d, tag in entities:
            bbox = np.array(gmsh.model.occ.getBoundingBox(d, tag))
            mins = np.minimum(mins, bbox[0:3])
            maxs = np.maximum(maxs, bbox[3:6])
        bounding_box = np.concatenate((mins, maxs))
        # For a single object, we could also just do:
        # bounding_box = np.array(list(gmsh.model.occ.get_bounding_box(dim, 1)))
    if dim == 2:
        bounding_box = bounding_box[[0, 1, 3, 4]]
    return bounding_box


def save_geometry(geom_fn: Callable, save_name: str) -> None:
    """
    Saves the geometry to a file.

    Args:
        geom_fn (Callable): Geometry function.
        save_name (Path): Path to save the geometry file.
    """
    gmsh.initialize()
    geom = geom_fn()
    gmsh.model.occ.synchronize()
    gmsh.write(save_name)
    gmsh.finalize()
