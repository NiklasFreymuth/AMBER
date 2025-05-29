from typing import Optional

from src.algorithms.amber.mesh_wrapper import MeshWrapper


def get_reconstructed_mesh(
        reference_mesh: MeshWrapper,
        sizing_field_query_scope: str = "elements",
        gmsh_kwargs: Optional[dict] = None,
) -> MeshWrapper:
    """
    Calculate a reference mesh from an expert mesh. This is done by calculating a sizing field on the expert mesh,
    and then building a new mesh from this sizing field. This is essentially an upper bound on the mesh quality
    that we can achieve when we want to reconstruct a mesh from a sizing field.
    Args:
        reference_mesh:
        sizing_field_query_scope: The type of sizing field to use. Can be "elements" or "nodes"
        gmsh_kwargs: Additional keyword arguments for the Gmsh mesh generation algorithm

    Returns: A mesh that is reconstructed from the sizing field of the expert mesh

    """
    from src.algorithms.amber.amber_util import get_sizing_field
    from src.environments.domains.gmsh_util import update_mesh

    sizing_field = get_sizing_field(mesh=reference_mesh, sizing_field_query_scope=sizing_field_query_scope)
    reconstructed_mesh = update_mesh(old_mesh=reference_mesh.mesh, sizing_field=sizing_field, gmsh_kwargs=gmsh_kwargs)
    return reconstructed_mesh
