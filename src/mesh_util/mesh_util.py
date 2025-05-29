import numpy as np
from skfem import Basis, ElementTetP1, ElementTriP1, Mesh

from src.tasks.domains import ExtendedMeshTet1, ExtendedMeshTri1
from src.tasks.domains.mesh_wrapper import MeshWrapper


def get_element_midpoints(mesh: Mesh, transpose: bool = True) -> np.array:
    """
    Get the midpoint of each element
    Args:
        mesh: The mesh as a skfem.Mesh object
        transpose: Whether to transpose the result. If True
            the result will be of shape (num_elements, 2), if False, it will be of shape (2, num_elements). Defaults
            to True.

    Returns: Array of shape (num_elements, 2)/(2, num_elements) containing the midpoint of each element

    """
    midpoints = np.mean(mesh.p[:, mesh.t], axis=1)
    if transpose:
        return midpoints.T
    else:
        return midpoints


def get_aggregation_per_element(
    solution: np.array,
    element_indices: np.array,
    aggregation_function_str: str = "mean",
) -> np.array:
    """
    get aggregation of solution per element from solution per vertex by adding all spanning vertices for each element

    Args:
        solution: Error estimate per element of shape (num_elements, solution_dimension)
        element_indices: Elements of the mesh. Array of shape (num_elements, vertices_per_element),
        where vertices_per_element is 3 triangular meshes
        aggregation_function_str: The aggregation function to use. Can be 'mean', 'std', 'min', 'max', 'median'
    Returns: An array of shape (num_elements, ) containing the solution per element

    """
    if aggregation_function_str == "mean":
        solution_per_element = solution[element_indices].mean(axis=1)
    elif aggregation_function_str == "std":
        solution_per_element = solution[element_indices].std(axis=1)
    elif aggregation_function_str == "min":
        solution_per_element = solution[element_indices].min(axis=1)
    elif aggregation_function_str == "max":
        solution_per_element = solution[element_indices].max(axis=1)
    elif aggregation_function_str == "median":
        solution_per_element = np.median(solution[element_indices], axis=1)
    else:
        raise ValueError(f"Aggregation function {aggregation_function_str} not supported")
    return solution_per_element


def to_scalar_basis(basis, linear: bool = False) -> Basis:
    """
    Returns: The scalar basis used for the error estimation via integration.

    """
    if linear:
        if basis.mesh.dim() == 2:
            element = ElementTriP1()
        elif basis.mesh.dim() == 3:
            element = ElementTetP1()
        else:
            raise ValueError(f"Unknown basis dimension {basis.mesh.dim()}")
    else:
        element = basis.elem
        while hasattr(element, "elem"):
            element = element.elem
    scalar_basis = basis.with_element(element)
    return scalar_basis


def probes_from_elements(basis: Basis, x: np.ndarray, cells: np.ndarray):
    """
    Return matrix which acts on a solution vector to find its values. Uses pre-computed cell indices.
    on points `x`.
    Args:
        basis: The basis to use for the interpolation
        x: The points to interpolate to
        cells: Cell indices per point. A cell index of -1 means that the point is not in any cell.

    Returns:

    """
    import sys

    if "pyodide" in sys.modules:
        from scipy.sparse.coo import coo_matrix
    else:
        from scipy.sparse import coo_matrix
    pts = basis.mapping.invF(x[:, :, np.newaxis], tind=cells)
    phis = np.array([basis.elem.gbasis(basis.mapping, pts, k, tind=cells)[0] for k in range(basis.Nbfun)]).flatten()
    return coo_matrix(
        (
            phis,
            (
                np.tile(np.arange(x.shape[1]), basis.Nbfun),
                basis.element_dofs[:, cells].flatten(),
            ),
        ),
        shape=(x.shape[1], basis.N),
    )


def get_volume_from_bounding_box(*, bounding_box: np.ndarray) -> float:
    """
    Computes the volume of a provided bounding box enclosing an underlying geometry.

    The bounding volume is calculated as the product of the extents in each dimension,.

    Args:
        bounding_box (np.ndarray): Bounding box of the geometry, represented as an array of shape (2*dim)

    Returns:
        float: The computed bounding volume of the mesh.

    """
    bounding_box = bounding_box.reshape(2, -1)
    volume = np.prod(np.abs(bounding_box[0] - bounding_box[1]))
    return volume


def get_longest_side_from_bounding_box(bounding_box: np.ndarray) -> float:
    """
    Computes the longest side length of the bounding box enclosing the given geometry.

    The longest side is determined by finding the maximum extent among all dimensions,
    using the minimum and maximum coordinates of the bounding box.

    Args:
        bounding_box (np.ndarray): Bounding box of the geometry, represented as an array of shape (2*dim)

    Returns:
        float: The length of the longest side of the bounding box.
    """
    bounding_box = bounding_box.reshape(2, -1)
    longest_side = np.max(np.abs(bounding_box[0] - bounding_box[1]))
    return longest_side


def get_midpoint_correspondences(from_mesh: MeshWrapper, to_mesh: MeshWrapper) -> np.ndarray:
    """
    Create a mapping from the midpoints of mesh1 to those of mesh2.
    This is done by finding for each element in mesh1 the element in mesh2 that contains its midpoints.
    As there may be differences in the mesh boundaries between meshes, we map midpoints in one mesh that are not part
    of the other to the nearest element in terms of Euclidean distance of the midpoints

    Args:
        to_mesh: A (coarse) MeshWrapper object. Has N elements
        from_mesh: A (fine) MeshWrapper object. Has M elements

    Returns: A numpy array of shape (N, ) containing indices in range(M) that specify
        the elements of mesh2 that contain the midpoints of the elements of mesh1,
        or the closest elements if there is no containment due to a mismatch in geometry.
    """
    mesh1_midpoints = from_mesh.element_midpoints
    return to_mesh.find_closest_elements(mesh1_midpoints)


def compute_boundary_vertex_normals(mesh: ExtendedMeshTet1 | ExtendedMeshTri1) -> np.ndarray:
    """
    Compute averaged outward unit normals at boundary nodes of a MeshTet1.

    Parameters
    ----------
    mesh : skfem.MeshTet1
        A tetrahedral mesh from scikit-fem.

    Returns
    -------
    boundary_nodes : np.ndarray of shape (n_boundary_nodes,)
        Indices of the boundary nodes in the mesh.

    boundary_node_normals : np.ndarray of shape (3, n_boundary_nodes)
        Unit normal vectors averaged at each boundary node.
    """
    # Get boundary facets
    boundary_facets = mesh.boundary_facets()

    # Get the mapping object
    mapping = mesh.mapping()

    # Compute normals at the center of each boundary facet
    # xi is the reference coordinate; for center, use zeros
    xi = np.zeros((mesh.dim() - 1, 1))
    tind = mesh.f2t[0, boundary_facets]
    normals = mapping.normals(xi, tind, boundary_facets, mesh.t2f).squeeze()

    # Facet-to-node mapping
    facet_nodes = mesh.facets[:, boundary_facets]  # shape (K, M), K=3 for triangles

    # Repeat normals for each node in the facet
    repeated_normals = np.repeat(normals, facet_nodes.shape[0], axis=1)  # shape (3, K*M)
    flat_nodes = facet_nodes.T.reshape(-1)  # shape (K*M,)

    # Accumulate normals per node
    node_normals = np.zeros((mesh.dim(), mesh.p.shape[1]))
    np.add.at(node_normals, (slice(None), flat_nodes), repeated_normals)

    # Count number of contributions per node
    node_counts = np.bincount(flat_nodes, minlength=mesh.p.shape[1])

    # Normalize accumulated normals
    nonzero = node_counts > 0
    node_normals[:, nonzero] /= node_counts[nonzero]

    norms = np.linalg.norm(node_normals[:, nonzero], axis=0)
    node_normals[:, nonzero] /= norms

    # Restrict to boundary nodes
    boundary_nodes = mesh.boundary_nodes()
    boundary_node_normals = node_normals[:, boundary_nodes]

    return boundary_node_normals


def compute_curvature(mesh: ExtendedMeshTet1, boundary_normals: np.ndarray) -> np.ndarray:
    """
    Compute signed curvature at boundary edges of a 3D triangular mesh.

    Curvature is calculated from the angle between boundary vertex normals at
    each edge and is signed based on the divergence of the edge vectors
    when boundary vertices are moved along their normals.

    Parameters
    ----------
    boundary_normals : np.ndarray of shape (3, n_boundary_vertices)
    mesh : ExtendedMeshTet1
        A mesh object that provides access to vertex positions, boundary nodes,
        boundary edges, and boundary vertex normals.

    Returns
    -------
    np.ndarray
        A 1D array of signed curvature values for each boundary edge.

    Notes
    -----
    The sign convention is as follows:
      - Positive curvature (convex): outward bending
      - Negative curvature (concave): inward bending

    ASCII diagram (side view):

              Normal ↑
                     |
        convex       |         concave
                     |
            *----*   |     *----*
           /     \   |     \   /
          *       *  |      * *
                     |
    """

    # Boundary vertex positions: shape (3, N_boundary_vertices)
    boundary_nodes = mesh.boundary_nodes()
    boundary_vertices = mesh.p.T[boundary_nodes].T

    mesh_scale = np.max((mesh.p.max(axis=1) - mesh.p.min(axis=1)))

    if mesh.dim() == 2:
        edges = mesh.facets[:, mesh.boundary_facets()]
    else:
        # 3D case: use edges
        edges = mesh.edges[:, mesh.boundary_edges()]
    # Boundary edges: shape (2, N_boundary_edges)

    # Map global vertex indices to local boundary vertex indices
    assert np.all(boundary_nodes[:-1] <= boundary_nodes[1:]), "boundary_nodes must be sorted"
    edge_indices = np.searchsorted(boundary_nodes, edges)

    # Normals at edge endpoints
    n0 = boundary_normals[:, edge_indices[0]]  # shape (3, M)
    n1 = boundary_normals[:, edge_indices[1]]  # shape (3, M)

    # Angle between normals using dot product
    dot = np.sum(n0 * n1, axis=0)
    dot = np.clip(dot, -1.0, 1.0)
    angle = np.arccos(dot)  # shape (M,)

    # Edge vectors
    v0 = boundary_vertices[:, edge_indices[0]]  # shape (3, M)
    v1 = boundary_vertices[:, edge_indices[1]]  # shape (3, M)
    edge_vector = v1 - v0

    # Move vertices along normals by a (relatively small) distance
    moved_v0 = v0 + n0 * 0.001 * mesh_scale
    moved_v1 = v1 + n1 * 0.001 * mesh_scale
    moved_edge_vector = moved_v1 - moved_v0

    # Compare original and moved edge lengths to assign curvature sign
    original_length = np.linalg.norm(edge_vector, axis=0)
    moved_length = np.linalg.norm(moved_edge_vector, axis=0)
    sign = -np.sign(original_length - moved_length)

    # Signed curvature
    signed_curvature = sign * angle  # shape (M,)
    return signed_curvature
