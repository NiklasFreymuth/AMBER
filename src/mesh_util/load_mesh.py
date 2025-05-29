from typing import Optional

import meshio
from skfem import MeshTet, MeshTri
from skfem.io import from_meshio

from src.tasks.domains.extended_mesh_tet1 import ExtendedMeshTet1
from src.tasks.domains.extended_mesh_tri1 import ExtendedMeshTri1


def load_expert_mesh(expert_mesh_path: str, extension: Optional[str] = None) -> ExtendedMeshTri1 | ExtendedMeshTet1:
    if extension is not None and not expert_mesh_path.endswith(f".{extension}"):
        expert_mesh_path = f"{expert_mesh_path}.{extension}"
    msh = meshio.read(expert_mesh_path)
    mesh = from_meshio(msh)
    vertex_positions = mesh.p
    if isinstance(mesh, MeshTri):
        mesh = ExtendedMeshTri1(vertex_positions, mesh.t)
    elif isinstance(mesh, MeshTet):
        mesh = ExtendedMeshTet1(vertex_positions, mesh.t)
    else:
        raise ValueError(f"Unsupported mesh type {type(mesh)}")
    return mesh
