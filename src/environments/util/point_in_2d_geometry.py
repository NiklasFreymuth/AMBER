from typing import Optional

import numpy as np
from numba import njit, prange


@njit(fastmath=True)
def points_on_edges(points: np.array, edges: np.array, candidate_indices: Optional[np.array] = None):
    """
    Determines which edge each point lies on. If the point does not lie on any edge, the index is -1.
    Args:
        points: An array of points with shape (2, N)
        edges: An array of edges with shape (M, 2, 2), where each edge is defined by its start and end points.
        candidate_indices: An array of triangle indices with shape (N, K) containing K candidate edges per queried point
    Returns: An array of edge indices with shape (num_points, ) containing the index of the edge that contains the
        point. If the point does not lie on any edge, the index is -1.

    """
    points_on_edges = np.empty(points.shape[0], dtype=np.int64)
    points_on_edges.fill(-1)  # -1 means no triangle found for this point
    for point_index, point in enumerate(points):
        if candidate_indices is not None:
            edge_indices = candidate_indices[point_index].astype(np.int64)
        else:
            edge_indices = np.arange(edges.shape[0], dtype=np.int64)
        for edge_index in edge_indices:
            current_edge = edges[edge_index]
            # the commented out code is equivalent to the following, but roughly 100x slower due to numpy array accesses
            # edge_vector = current_edge[1] - current_edge[0]
            # point_vector = point - current_edge[0]
            # cross_product = point_vector[1] * edge_vector[0] - point_vector[0] * edge_vector[1]
            # dot_product = point_vector[0] * edge_vector[0] + point_vector[1] * edge_vector[1]
            # squared_length = edge_vector[0] * edge_vector[0] + edge_vector[1] * edge_vector[1]
            # edge_vector = current_edge[1] - current_edge[0]
            # point_vector = point - current_edge[0]

            abs_cross_product = np.abs(
                np.subtract(
                    np.multiply(
                        np.subtract(point[1], current_edge[0, 1]),
                        np.subtract(current_edge[1, 0], current_edge[0, 0]),
                    ),
                    np.multiply(
                        np.subtract(point[0], current_edge[0, 0]),
                        np.subtract(current_edge[1, 1], current_edge[0, 1]),
                    ),
                )
            )
            dot_product = np.add(
                np.multiply(
                    np.subtract(point[0], current_edge[0, 0]),
                    np.subtract(current_edge[1, 0], current_edge[0, 0]),
                ),
                np.multiply(
                    np.subtract(point[1], current_edge[0, 1]),
                    np.subtract(current_edge[1, 1], current_edge[0, 1]),
                ),
            )
            squared_length = np.add(
                np.multiply(
                    np.subtract(current_edge[1, 0], current_edge[0, 0]),
                    np.subtract(current_edge[1, 0], current_edge[0, 0]),
                ),
                np.multiply(
                    np.subtract(current_edge[1, 1], current_edge[0, 1]),
                    np.subtract(current_edge[1, 1], current_edge[0, 1]),
                ),
            )
            if (abs_cross_product < 1e-12) + (0 <= dot_product) + (dot_product <= squared_length) == 3:
                points_on_edges[point_index] = edge_index
                break
    return points_on_edges


@njit(fastmath=True)
def points_in_triangles(
    points: np.array, triangles: np.array, candidate_indices: Optional[np.array] = None
) -> np.array:
    """
    Args:
        points (np.ndarray): array of points with shape (N, 2)
        triangles (np.ndarray): array of triangles. Shape (M, 3, 2)
        candidate_indices (np.ndarray): array of candidate triangle indices for each point. Shape (N, K), where
        K is the number of candidate triangles for each point. If None, all triangles are checked for each point.

    Returns:
        point_in (np.ndarray): Array with triangle index for each point. Shape (N, ) containing the index of the
        triangle that contains the point. If the point is not in any triangle, the index is -1.
    """
    N = points.shape[0]
    M = triangles.shape[0]

    point_in = np.full(N, -1, dtype=np.int64)

    # Pre-compute triangle edges
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 1]
    edge3 = triangles[:, 0] - triangles[:, 2]

    for point_index in range(N):
        point = points[point_index]

        if candidate_indices is not None:
            triangle_indices = candidate_indices[point_index]
        else:
            triangle_indices = np.arange(M, dtype=np.int64)

        for triangle_index in triangle_indices:
            e1 = edge1[triangle_index]
            e2 = edge2[triangle_index]
            e3 = edge3[triangle_index]

            b1 = np.sign(
                (point[0] - triangles[triangle_index, 1, 0]) * e1[1]
                - e1[0] * (point[1] - triangles[triangle_index, 1, 1])
            )
            b2 = np.sign(
                (point[0] - triangles[triangle_index, 2, 0]) * e2[1]
                - e2[0] * (point[1] - triangles[triangle_index, 2, 1])
            )
            b3 = np.sign(
                (point[0] - triangles[triangle_index, 0, 0]) * e3[1]
                - e3[0] * (point[1] - triangles[triangle_index, 0, 1])
            )

            if (np.abs(b1 + b2 + b3) + ((b1 == 0) + (b2 == 0) + (b3 == 0))) == 3:
                point_in[point_index] = triangle_index
                break

    return point_in


@njit(fastmath=True, parallel=True)
def parallel_points_in_triangles(
    points: np.array, triangles: np.array, candidate_indices: Optional[np.array] = None
) -> np.array:
    """
    Args:
        points (np.ndarray): array of points with shape (N, 2)
        triangles (np.ndarray): array of triangles. Shape (M, 3, 2)
        candidate_indices (np.ndarray): array of candidate triangle indices for each point. Shape (N, K), where
        K is the number of candidate triangles for each point. If None, all triangles are checked for each point.

    Returns:
        point_in (np.ndarray): Array with triangle index for each point. Shape (N, ) containing the index of the
        triangle that contains the point. If the point is not in any triangle, the index is -1.
    """
    N = points.shape[0]
    M = triangles.shape[0]

    # Declare the array outside the parallel loop
    point_in = np.empty(N, dtype=np.int64)
    point_in.fill(-1)

    # Use prange for parallelization
    for point_index in prange(N):
        point = points[point_index]

        if candidate_indices is not None:
            triangle_indices = candidate_indices[point_index]
        else:
            triangle_indices = np.arange(M)

        for triangle_index in triangle_indices:
            current_triangle = triangles[triangle_index]

            b1 = np.sign(
                (point[0] - current_triangle[1, 0]) * (current_triangle[0, 1] - current_triangle[1, 1])
                - (current_triangle[0, 0] - current_triangle[1, 0]) * (point[1] - current_triangle[1, 1])
            )

            b2 = np.sign(
                (point[0] - current_triangle[2, 0]) * (current_triangle[1, 1] - current_triangle[2, 1])
                - (current_triangle[1, 0] - current_triangle[2, 0]) * (point[1] - current_triangle[2, 1])
            )

            b3 = np.sign(
                (point[0] - current_triangle[0, 0]) * (current_triangle[2, 1] - current_triangle[0, 1])
                - (current_triangle[2, 0] - current_triangle[0, 0]) * (point[1] - current_triangle[0, 1])
            )

            if (np.abs(b1 + b2 + b3) + ((b1 == 0) + (b2 == 0) + (b3 == 0))) == 3:
                point_in[point_index] = triangle_index
                break

    return point_in


##################
# Test functions #
##################


def _generate_candidate_indices(points, triangles, k=5):
    """Generate an array of candidate triangle indices for each point using KDTree."""
    from pykdtree.kdtree import KDTree

    triangle_centers = np.mean(triangles, axis=1)
    tree = KDTree(triangle_centers)
    _, candidate_indices = tree.query(points, k)
    return candidate_indices


def _test_point_on_edges(use_candidate_indices: bool = True) -> None:
    import time

    from skfem import MeshTri

    np.random.seed(0)
    point_mesh = MeshTri.init_symmetric().refined(9)
    # query_points = point_mesh.p
    query_points = np.mean(point_mesh.p[:, point_mesh.t].T, axis=1)

    mesh = MeshTri()
    for i in range(10):
        mesh = mesh.refined(np.where(np.random.rand(mesh.nelements) < 0.5)[0])

    triangles = mesh.p[:, mesh.t].T
    edges = mesh.facets  # Assuming this gives the edge indices
    edge_positions = mesh.p[:, edges]
    print()
    print(f"  Query points: {query_points.shape}")  # , Query Mesh: {point_mesh},  Mesh: {mesh}")
    print(f"  Use candidate indices: {use_candidate_indices}")

    if use_candidate_indices:
        candidate_indices = _generate_candidate_indices(query_points, triangles).astype(np.int64)
        triangle_elements = parallel_points_in_triangles(query_points, triangles, candidate_indices)
        candidate_triangle_indices = mesh.t2f.T[triangle_elements]

    else:
        candidate_triangle_indices = None

    previous_solution = None
    for fn in (points_on_edges,):
        start = time.perf_counter()
        edge_membership = fn(
            points=query_points,
            edges=edge_positions.T,
            candidate_indices=candidate_triangle_indices,
        )
        done = time.perf_counter()

        if previous_solution is not None:
            assert np.all(edge_membership == previous_solution), f"Function name {fn.__name__} has wrong result :("
        previous_solution = edge_membership

        print(f"  Runtime: {done - start:.4f}     Fn {fn.__name__}")
        print(f"  Points on edges: {sum(edge_membership != -1)} / {query_points.shape[0]}")


def _test_point_in_triangle(use_candidate_indices: bool = True, num_points: int = 100000, max_refs: int = 7):
    import time

    from skfem import MeshTri

    np.random.seed(0)
    points = np.random.uniform((0.0, 0.0), (1.0, 1.0), size=(num_points, 2))
    for i in range(3, max_refs + 1):
        mesh = MeshTri.init_symmetric().refined(int(i))
        triangles = mesh.p[:, mesh.t].T

        if use_candidate_indices:
            candidate_indices = _generate_candidate_indices(points, triangles).astype(np.int64)
        else:
            candidate_indices = None

        print(f"Num Refinements: {i}, Use candidate indices: {use_candidate_indices}")

        previous_solution = None
        for fn in (
            points_in_triangles,
            parallel_points_in_triangles,
        ):
            start = time.perf_counter()
            point_in = fn(points, triangles, candidate_indices)
            done = time.perf_counter()

            if previous_solution is not None:
                assert np.all(point_in == previous_solution), f"Function name {fn.__name__} has wrong result :("
            previous_solution = point_in

            print(f"Runtime: {done - start:.4f}     Fn {fn.__name__}")


def main():
    _test_point_on_edges(use_candidate_indices=True)
    # _test_point_on_edges(use_candidate_indices=False)

    # _test_point_in_triangle(use_candidate_indices=True, num_points=10000000, max_refs=10)
    # _test_point_in_triangle(use_candidate_indices=True, num_points=100000, max_refs=12)
    # _test_point_in_triangle(use_candidate_indices=False, num_points=100000, max_refs=7)

    # colors = ["red", "green", "blue", "yellow"]
    # point_colors = []
    # for index in point_in[:1000]:
    #     point_colors.append(colors[index])

    # plt.scatter(points[:1000, 0], points[:1000, 1], color=point_colors)
    # for index, triangle in enumerate(triangles):
    #     plt.plot(triangle[:, 0], triangle[:, 1], color=colors[index])
    # plt.show()


if __name__ == "__main__":
    main()
