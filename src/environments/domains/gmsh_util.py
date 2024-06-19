from typing import Dict, Optional, Type

import gmsh
import numpy as np
import pygmsh
from skfem import Mesh
from skfem.io import from_meshio

from src.algorithms.amber.mesh_wrapper import MeshWrapper


class gmsh_session:
    """
    A context manager for a Gmsh session. This ensures that the Gmsh session is properly initialized and finalized.
    """

    def __init__(self, gmsh_kwargs: Optional[Dict] = None):
        if gmsh_kwargs is None:
            gmsh_kwargs = {}
        self.gmsh_kwargs = gmsh_kwargs

    def __enter__(self):
        gmsh.initialize()

        # Set default options
        gmsh.option.setNumber("General.Verbosity", 0)  # turn off info messages/prints
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Optimize", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", self.gmsh_kwargs.get("min_sizing_field", 1e-6))
        gmsh.option.setNumber("Mesh.MeshSizeMax", 10)

        # Set up a default model. We only ever use one model at a time, so the name "model" is fine.
        gmsh.model.add("model")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Finalize the Gmsh session
        gmsh.model.remove()
        gmsh.finalize()

        # Handle exceptions if necessary
        if exc_type:
            print(f"An exception occurred: {exc_val}")
        # Return False to propagate exceptions, True to suppress them
        return False


def square_hole_geom(hole_position: np.array, hole_size: np.array, length: float = 1.0):
    # Manually create a geometry instance
    geom = pygmsh.occ.Geometry()

    # Add rectangle and disk (hole) to the geometry
    rectangle = geom.add_rectangle([0.0, 0.0, 0.0], a=length, b=length)

    # Perform boolean difference
    hole = geom.add_disk([hole_position[0], hole_position[1], 0.0], hole_size)
    geom.boolean_difference(rectangle, hole)
    return geom


def sloped_l_geom(
        hole_position: np.array,
        hole_size: float,
        length: float = 1.0,
        cutout_length: float = 0.5,
):
    geom = pygmsh.occ.Geometry()

    # Create the outer rectangle (large part of the L-shape)
    outer_rectangle = geom.add_rectangle([0.0, 0.0, 0.0], a=length, b=length)

    # Create the inner rectangle (cutout to form the L-shape)
    # Assuming the cutout is from the top right corner, adjust as needed
    cutout_rectangle = geom.add_rectangle([length - cutout_length, 0.0, 0.0], a=cutout_length, b=length - cutout_length)

    # Subtract the inner rectangle from the outer to get the L-shape
    l_shape = geom.boolean_difference([outer_rectangle], [cutout_rectangle], delete_first=True, delete_other=True)

    # Add the disk to the geometry
    disk = geom.add_disk([hole_position[0], hole_position[1], 0.0], hole_size)

    # blend the disk with the L-shape
    final_shape = geom.boolean_union([l_shape, disk])

    return geom


def polygon_geom(
        boundary_nodes: np.array,
):
    """
    Creates a mesh from a polygon defined by its boundary nodes
    Args:
        boundary_nodes: Boundary nodes of the polygon. Has shape (num_nodes, 2)

    Returns: A mesh of the given class

    """
    geom = pygmsh.occ.Geometry()
    geom.add_polygon(boundary_nodes)
    return geom


def step_file_geom(step_file_path: str) -> pygmsh.geo.Geometry:
    """
    Load a step file and return a pygmsh geometry object representing the geometry of this file
    Args:
        step_file_path: (relative or absolute) path to the step file

    Returns: A pygmsh geometry object representing the geometry of the step file

    """
    # Initialize gmsh session
    gmsh.initialize()

    # Create a pygmsh geometry object
    geom = pygmsh.occ.Geometry()
    geom.import_shapes(step_file_path)

    return geom


def generate_initial_mesh(
        geometry_function: callable,
        desired_element_size: float,
        target_class: Optional[Type[Mesh]] = None,
        gmsh_kwargs: Optional[Dict] = None,
        dim: int = 2,
        verbose: bool = False,
) -> Mesh:
    with gmsh_session(gmsh_kwargs):
        geom = geometry_function()
        gmsh.model.occ.synchronize()  # Synchronize the geometry
        geom.characteristic_length_max = desired_element_size
        geom.characteristic_length_min = 0.7 * desired_element_size
        m = geom.generate_mesh(dim=dim, verbose=verbose)
        bounding_box = np.array(list(gmsh.model.occ.get_bounding_box(dim, 1)))

    mesh = from_meshio(m)  # Convert to scikit-fem mesh
    if target_class is not None:
        mesh = target_class(mesh.p, mesh.t)

    if dim == 2:
        bounding_box = bounding_box[[0, 1, 3, 4]]
    mesh.geom_bounding_box = bounding_box
    mesh.geom_fn = geometry_function  # store the geometry for later use.
    return mesh


def update_mesh(old_mesh: Mesh, sizing_field: np.ndarray,
                algorithm_idx: Optional[int] = None,
                gmsh_kwargs: Optional[dict] = None,
                sizing_field_positions: Optional[np.ndarray] = None) -> MeshWrapper:
    """
    Update the mesh using the given sizing field. This is done by creating a temporary file with the sizing field
    evaluations and using Gmsh to create a new mesh with the given sizing field.
    Args:
        old_mesh: Old mesh to update. Must have a geometry function set.
        sizing_field: Sizing field to use for the update. Has shape (num_elements {, 3})
        algorithm_idx: Index of the algorithm to use for the mesh generation. See Gmsh documentation for details.
        gmsh_kwargs: Optional keyword arguments to pass to the Gmsh mesh generation function
        sizing_field_positions: Optional node positions to use for the sizing field generation. If provided, will use
         these node positions. Else, will use the vertices of the mesh
    """
    import os
    import tempfile

    assert hasattr(old_mesh, "geom_fn"), "Geometry function not set"

    dimension = old_mesh.dim()
    geometry_function = old_mesh.geom_fn  # function to create the geometry

    if sizing_field_positions is None:
        sizing_field_positions = old_mesh.p[:, old_mesh.t].T

    assert len(sizing_field) == len(sizing_field_positions), (
        f"Sizing field has wrong shape. Given {sizing_field.shape}, expected {sizing_field.shape}"
    )

    if algorithm_idx is None:
        algorithm_idx = 6  # default to Delaunay
        # use, e.g., 9 for nicely aligned triangles

    # Create a temporary file using tempfile, which automatically handles the file location
    tmpfile = tempfile.NamedTemporaryFile(delete=False)
    write_sizing_field_to_tmpfile(sizing_field_positions=sizing_field_positions,
                                  sizing_field=sizing_field,
                                  tmpfile=tmpfile)
    tmpfile_path = tmpfile.name
    tmpfile.close()

    # generate the new mesh
    with gmsh_session(gmsh_kwargs):
        field_idx = 1
        geom = geometry_function()
        gmsh.model.occ.synchronize()  # Synchronize the geometry
        gmsh.merge(tmpfile_path)
        gmsh.model.mesh.field.add("PostView", field_idx)
        gmsh.model.mesh.field.setNumber(field_idx, "ViewIndex", 0)
        gmsh.model.mesh.field.setAsBackgroundMesh(field_idx)
        m = geom.generate_mesh(dim=dimension, verbose=False, algorithm=algorithm_idx)

    # clean up and convert
    os.remove(tmpfile_path)
    mesh = from_meshio(m)  # Convert to scikit-fem mesh
    mesh = old_mesh.__class__(mesh.p, mesh.t)  # Reset class to the original (potentially custom) class
    mesh.geom_fn = geometry_function  # Reset the geometry function
    mesh.geom_bounding_box = old_mesh.geom_bounding_box
    mesh = MeshWrapper(mesh)
    return mesh


def write_sizing_field_to_tmpfile(
        sizing_field_positions: np.ndarray, sizing_field: np.ndarray, tmpfile, chunk_size: int = 10000
):
    """
    Write the sizing field evaluations to a temporary file in the .pos format.
    This format is used by Gmsh to define a sizing field. Here, every line defines a (triangular/tetrahedral)
    mesh element and the corresponding sizing field evaluation on the nodes. We use the gmsh .pos formats as described
    on page 131 of this document: https://gmsh.info/dev/doc/texinfo/gmsh.pdf
        For triangles, we use the format ST(x1,y1,z1,x2,y2,z2,x3,y3,z3){s1,s2,s3}; (for Scalar Triangle) where
        (x1,y1,z1), (x2,y2,z2), (x3,y3,z3) are the coordinates of the nodes of the element and
        (s1,s2,s3) are the evaluations of the sizing field at these nodes. In 2d, the z-coordinates are set to 0.
        For tetrahedra, we use the format SS(x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4){s1,s2,s3,s4}; (for Scalar tetrahedron)
        where (x1,y1,z1), (x2,y2,z2), (x3,y3,z3), (x4,y4,z4) are the coordinates of the nodes of the element and
        (s1,s2,s3,s4) are the evaluations of the sizing field at these (now 4) nodes.
    Args:
        sizing_field_positions: Positions of the mesh nodes/vertices to span the sizing field over.
            For simplical elements, this has shape num_nodes, dim+1, dim), where
                the first dimension is the node index,
                the second dimension is the vertex index of the simplex, and
                the third dimension is the spatial dimension.
            For simple vertex positions/pixels/voxels, this has shape (num_nodes, dim), where
                the first dimension is the node index and
                the second dimension is the spatial dimension (x, y{, z})
        sizing_field: Evaluations of the sizing field at the mesh nodes. May have shape (num_elements,) or shape
            (num_elements, dim+1). If the shape is (num_elements, ), each field will be broadcast to its simplex's
            vertices.
        tmpfile: tmpfile object to write to
        chunk_size: Size of the chunks to write to the file at once. This is useful for large meshes to avoid memory
            issues.

    Returns: None

    """
    assert sizing_field_positions.ndim in [2, 3], "node_positions must be 2D or 3D"
    if sizing_field_positions.shape[-1] == 2:  # 2d positions. Broadcast to z=0
        sizing_field_positions = np.concatenate(
            (sizing_field_positions, np.zeros(sizing_field_positions.shape[:-1] + (1,))), axis=-1)

    if sizing_field_positions.ndim == 2:  # picture elements/vertices
        assert sizing_field.ndim == 1, "Sizing field must have shape (num_elements,) when only providing points"
        lines = [
            f"SP({pos[0]},{pos[1]},{pos[2]})"
            f"{{{size}}};"
            for pos, size in zip(sizing_field_positions, sizing_field)
        ]
    else:
        assert sizing_field_positions.shape[1] in [3, 4], \
            f"Unsupported node_positions shape {sizing_field_positions.shape}"
        if sizing_field.ndim == 1:
            # broadcast the sizing field to the vertices of the elements
            sizing_field = np.repeat(sizing_field[:, None], sizing_field_positions.shape[1], axis=1)

        # get a list of lines to write to the file
        if sizing_field_positions.shape[1] == 3:  # triangles
            lines = [
                f"ST({pos[0][0]},{pos[0][1]},{pos[0][2]},"
                f"{pos[1][0]},{pos[1][1]},{pos[1][2]},"
                f"{pos[2][0]},{pos[2][1]},{pos[2][2]})"
                f"{{{size[0]},{size[1]},{size[2]}}};"
                for pos, size in zip(sizing_field_positions, sizing_field)
            ]
        else:  # tetrahedra
            lines = [
                f"SS({pos[0][0]},{pos[0][1]},{pos[0][2]},"
                f"{pos[1][0]},{pos[1][1]},{pos[1][2]},"
                f"{pos[2][0]},{pos[2][1]},{pos[2][2]},"
                f"{pos[3][0]},{pos[3][1]},{pos[3][2]})"
                f"{{{size[0]},{size[1]},{size[2]},{size[3]}}};"
                for pos, size in zip(sizing_field_positions, sizing_field)
            ]

    _write_to_tmpfile(chunk_size, lines, tmpfile)


def _write_to_tmpfile(chunk_size, lines, tmpfile):
    # write to the file
    tmpfile.write(b'View "sizing_field" {\n')
    for i in range(0, len(lines), chunk_size):
        # write the lines to the file in chunks to avoid memory issues
        content = "\n".join(lines[i: i + chunk_size]) + "\n"
        tmpfile.write(content.encode("utf-8"))
    tmpfile.write(b"};\n")
