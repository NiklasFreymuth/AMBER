import os
import tempfile
from pathlib import Path

import gmsh
import meshio
from skfem import Mesh
from skfem.io.meshio import to_meshio

from src.tasks.domains.mesh_wrapper import MeshWrapper


def save_as_vtk(mesh: Mesh | MeshWrapper, output_vtk_path: str | Path, verbose: bool = False) -> None:
    """
    Convert a scikit FEM mesh to a legacy VTK file using Gmsh for format conversion.

    Args:
        mesh (meshio.Mesh): The input meshio mesh object.
        output_vtk_path (str): Path to the output VTK file.
    """
    if isinstance(output_vtk_path, Path):
        output_vtk_path = str(output_vtk_path)
    if isinstance(mesh, MeshWrapper):
        # Unwrap
        mesh = mesh.mesh

    mesh = to_meshio(mesh)
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Write the meshio mesh to a temporary .msh file
        meshio.write(tmp_path, mesh, file_format="gmsh22")

        # Use Gmsh to open the .msh and write a .vtk
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", int(verbose))  # turn off info messages/prints
        gmsh.option.setNumber("General.Verbosity", int(verbose))  # turn off info messages/prints
        gmsh.open(tmp_path)
        gmsh.write(output_vtk_path)
        gmsh.finalize()

    finally:
        os.remove(tmp_path)
