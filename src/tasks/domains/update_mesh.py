from typing import Optional, Union

import gmsh
import numpy as np
from skfem import Mesh
from skfem.io import from_meshio

from src.tasks.domains.gmsh_session import gmsh_session
from src.tasks.domains.mesh_wrapper import MeshWrapper


def update_mesh(
    old_mesh: Union[Mesh, MeshWrapper],
    sizing_field: np.ndarray,
    algorithm_idx: Optional[int] = None,
    gmsh_kwargs: Optional[dict] = None,
    sizing_field_positions: Optional[np.ndarray] = None,
) -> MeshWrapper:
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

    if isinstance(old_mesh, MeshWrapper):
        old_mesh = old_mesh.mesh

    dimension = old_mesh.dim()
    geometry_function = old_mesh.geom_fn  # function to create the geometry

    if sizing_field_positions is None:
        # unless otherwise specified, use the vertices of the mesh as the positions for the sizing field
        sizing_field_positions = old_mesh.p[:, old_mesh.t].T

    if len(sizing_field) == old_mesh.nvertices:
        # sizing field is on the mesh vertices. Broadcast to the elements
        sizing_field = sizing_field[old_mesh.t].T

    assert len(sizing_field) == len(
        sizing_field_positions
    ), f"Sizing field has wrong shape. Given {sizing_field.shape}, expected {sizing_field_positions.shape}"

    if algorithm_idx is None:
        algorithm_idx = 6  # default to Delaunay
        # use, e.g., 9 for nicely aligned triangles

    # Create a temporary file using tempfile, which automatically handles the file location
    tmpfile = tempfile.NamedTemporaryFile(delete=False)
    write_sizing_field_to_tmpfile(sizing_field_positions=sizing_field_positions, sizing_field=sizing_field, tmpfile=tmpfile)
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
    mesh = old_mesh.convert_new_mesh(new_mesh=mesh)
    mesh = MeshWrapper(mesh)
    return mesh


def write_sizing_field_to_tmpfile(sizing_field_positions: np.ndarray, sizing_field: np.ndarray, tmpfile, chunk_size: int = 10000):
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
        sizing_field_positions = np.concatenate((sizing_field_positions, np.zeros(sizing_field_positions.shape[:-1] + (1,))), axis=-1)

    if sizing_field_positions.ndim == 2:  # picture elements/vertices
        assert sizing_field.ndim == 1, "Sizing field must have shape (num_elements,) when only providing points"
        lines = [f"SP({pos[0]},{pos[1]},{pos[2]})" f"{{{size}}};" for pos, size in zip(sizing_field_positions, sizing_field)]
    else:
        assert sizing_field_positions.shape[1] in [
            3,
            4,
        ], f"Unsupported node_positions shape {sizing_field_positions.shape}"
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
        content = "\n".join(lines[i : i + chunk_size]) + "\n"
        tmpfile.write(content.encode("utf-8"))
    tmpfile.write(b"};\n")
