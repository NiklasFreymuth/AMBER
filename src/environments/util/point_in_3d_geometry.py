from typing import Optional

import numpy as np
from numba import njit


@njit(fastmath=True)
def points_on_faces(
    points: np.array,
    faces: np.array,
    candidate_indices: Optional[np.array] = None,
    tolerance: float = 1e-12,
) -> np.array:
    """
    Args:
        points (np.ndarray): array of points with shape (N, 3)
        faces (np.ndarray): array of faces with shape (M, 3, 3)
        candidate_indices (np.ndarray): array of candidate face indices for each point. Shape (N, K),
        where K is the number of candidate faces for each point. If the point is known to lie inside a tetrahedron,
        these can be the faces of the tetrahedron. If None, all faces are checked for each point.

    Returns:
        point_on_face (np.ndarray): Array with face index for each point. Shape (N, ),
        If the point is not on any face, the index is -1.
    """
    N = points.shape[0]
    M = faces.shape[0]

    point_on_face = np.full(N, -1, dtype=np.int64)

    for point_index in range(N):
        point = points[point_index]
        if candidate_indices is not None:
            face_indices = candidate_indices[point_index]
        else:
            face_indices = np.arange(M)

        for face_index in face_indices:
            current_face = faces[face_index]
            A, B, C = current_face

            # Normal vector of the plane
            normal = np.cross(B - A, C - A)

            # Check if the point is in the plane
            dot_product = np.dot(normal, point - A)
            if -tolerance < dot_product and dot_product < tolerance:
                # Compute vectors
                v0 = C - A
                v1 = B - A
                v2 = point - A

                # Compute dot products
                dot00 = np.dot(v0, v0)
                dot01 = np.dot(v0, v1)
                dot02 = np.dot(v0, v2)
                dot11 = np.dot(v1, v1)
                dot12 = np.dot(v1, v2)

                # Compute barycentric coordinates
                inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
                u = (dot11 * dot02 - dot01 * dot12) * inv_denom
                v = (dot00 * dot12 - dot01 * dot02) * inv_denom

                # Check if point is in triangle
                if (u >= 0) and (v >= 0) and (u + v < 1):
                    point_on_face[point_index] = face_index
                    break  # Stop checking other faces if found

    return point_on_face


@njit(fastmath=True)
def points_in_tetrahedra(
    points: np.array,
    tetrahedra: np.array,
    candidate_indices: Optional[np.array] = None,
    tolerance: float = 1e-15,
) -> np.array:
    """
    Args:
        points (np.ndarray): array of points with shape (N, 3)
        tetrahedra (np.ndarray): array of tetrahedra with shape (M, 4, 3)
        candidate_indices (np.ndarray): array of candidate tetrahedron indices for each point. Shape (N, K), where
        K is the number of candidate tetrahedra for each point. If None, all tetrahedra are checked for each point.

    Returns:
        point_in (np.ndarray): Array with tetrahedron index for each point. Shape (N, ) containing the index of the
        tetrahedron that contains the point. If the point is not in any tetrahedron, the index is -1.
    """
    N = points.shape[0]
    M = tetrahedra.shape[0]

    # Declare the array
    point_in = np.empty(N, dtype=np.int64)
    point_in.fill(-1)

    # Standard loop
    for point_index in range(N):
        point = points[point_index]

        if candidate_indices is not None:
            tetrahedron_indices = candidate_indices[point_index]
        else:
            tetrahedron_indices = np.arange(M)

        for tetrahedron_index in tetrahedron_indices:
            current_tetrahedron = tetrahedra[tetrahedron_index]
            A, B, C, D = current_tetrahedron

            # Manually populate the matrix to avoid np.array or np.vstack
            mat = np.empty((3, 3))
            mat[:, 0] = A - D
            mat[:, 1] = B - D
            mat[:, 2] = C - D

            # Calculate barycentric coordinates
            mat_inv = np.linalg.inv(mat)
            bary_coords = np.dot(mat_inv, point - D)
            delta = 1 - np.sum(bary_coords)

            # Check if the point is inside the tetrahedron
            if (
                np.all(bary_coords >= -tolerance)
                and np.all(bary_coords <= 1 + tolerance)
                and (0 - tolerance <= delta <= 1 + tolerance)
            ):
                point_in[point_index] = tetrahedron_index
                break

    return point_in


##################
# Test functions #
##################


def _generate_candidate_indices(points, tetrahedra, k=5):
    """Generate an array of candidate triangle indices for each point using KDTree."""
    from pykdtree.kdtree import KDTree

    tetrahedra_centers = np.mean(tetrahedra, axis=1)
    tree = KDTree(tetrahedra_centers)
    _, candidate_indices = tree.query(points, k)
    return candidate_indices


def _test_point_in_tetrahedra(
    use_candidate_indices: bool = True,
    num_points: int = 100000,
    min_refs: int = 3,
    max_refs: int = 7,
    plot: bool = False,
):
    import time

    from skfem import MeshTet

    np.random.seed(0)
    points = np.random.uniform(0, 1, size=(num_points, 3))

    p = np.linspace(0, 1, 2)
    mesh = MeshTet.init_tensor(*(p,) * 3)
    for i in range(min_refs, max_refs + 1):
        tetrahedra = mesh.p[:, mesh.t].T

        if use_candidate_indices:
            candidate_indices = _generate_candidate_indices(points, tetrahedra).astype(np.int64)
        else:
            candidate_indices = None

        print(f"Num Refinements: {i}, Use candidate indices: {use_candidate_indices}")

        previous_solution = None
        for fn in (points_in_tetrahedra,):
            start = time.perf_counter()
            point_in = fn(points, tetrahedra, candidate_indices)
            done = time.perf_counter()

            print(f"Runtime: {done - start:.4f}     Fn {fn.__name__}")
            print(f"Missing Points: {np.sum(point_in == -1)}")

            if plot:
                _plot_points(points, point_in)
        mesh = mesh.refined(1)


def _plot_points(points, point_in):
    import plotly.graph_objects as go

    scatter_trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        showlegend=True,
        marker=dict(
            size=3,
            color=point_in,  # set color to an array/list of desired values
            colorscale="Viridis",  # choose a colorscale
            opacity=0.5,
            colorbar=dict(
                title="Idx",
                yanchor="top",
                y=1,
                x=0,
                ticks="outside",  # put colorbar on the left
            ),
        ),
        hovertemplate="v: %{marker.color:.8f}<br>" + "x: %{x:.8f}<br>y: %{y:.8f}<br>z: %{z:.8f}<extra></extra>",
        name="Value (v)",
    )
    # configure figure from trace
    fig = go.Figure(data=scatter_trace)
    fig.show()


def _test_simple_points_in_tetrahedra():
    # Define a single tetrahedron with vertices at (0, 0, 0), (1, 0, 0), (0, 1, 0), and (0, 0, 1)
    tetrahedra = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]])

    # Define some points
    points = np.array(
        [
            [0.1, 0.1, 0.1],  # Inside the tetrahedron
            [0.5, 0.5, 0.5],  # Outside the tetrahedron
            [0, 0, 0],  # On a vertex
        ]
    )

    # Manually determined indices: [0, -1, 0]
    expected_indices = np.array([0, -1, 0])

    # Calculate indices using the function
    calculated_indices = points_in_tetrahedra(points, tetrahedra)

    # Compare
    assert np.array_equal(
        expected_indices, calculated_indices
    ), f"Expected {expected_indices}, got {calculated_indices}"


def _test_point_on_faces():
    from skfem import MeshTet

    np.random.seed(0)
    points = np.random.uniform(0, 1, size=(1000, 3))

    p = np.linspace(0, 1, 2)
    mesh = MeshTet.init_tensor(*(p,) * 3)
    mesh = mesh.refined(1)
    tetrahedra = mesh.p[:, mesh.t].T

    faces = np.empty((tetrahedra.shape[0], 3, 3))
    faces[:, 0] = tetrahedra[:, 0]
    faces[:, 1] = tetrahedra[:, 1]
    faces[:, 2] = tetrahedra[:, 2]

    # draw 100 points per face by using a random convex combination of faces
    num_faces = faces.shape[0]
    num_points_per_face = 100
    points = np.empty((num_faces * num_points_per_face, 3))
    for i in range(num_faces):
        face = faces[i]
        alpha = np.random.uniform(0, 1, size=(num_points_per_face, 3))
        alpha = alpha / np.sum(alpha, axis=1)[:, None]
        points[i * num_points_per_face : (i + 1) * num_points_per_face] = np.einsum("ij, jk->ik", alpha, face)

    points = points.astype(float)
    faces = faces.astype(float)

    point_on_face = points_on_faces(points, faces)
    print(f"Missing Points: {np.sum(point_on_face == -1)}")


def main():
    # _test_point_in_tetrahedra(use_candidate_indices=False, num_points=10000, max_refs=7, plot=False)
    # _test_point_in_tetrahedra(
    #     use_candidate_indices=False, num_points=100000, max_refs=3, plot=True
    # )
    # _test_simple_points_in_tetrahedra()
    _test_point_on_faces()


if __name__ == "__main__":
    main()
