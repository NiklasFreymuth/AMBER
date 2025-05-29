from typing import Optional

import numpy as np


def points_in_simplices(points: np.ndarray, simplices: np.ndarray, candidate_indices: Optional[np.ndarray] = None) -> np.ndarray:
    if points.shape[1] == 2:
        from src.mesh_util.point_in_2d_geometry import parallel_points_in_triangles

        return parallel_points_in_triangles(points=points, triangles=simplices, candidate_indices=candidate_indices)
    elif points.shape[1] == 3:
        from src.mesh_util.point_in_3d_geometry import points_in_tetrahedra

        return points_in_tetrahedra(points=points, tetrahedra=simplices, candidate_indices=candidate_indices)
    else:
        raise ValueError(f"Unsupported dimension: {points.shape[1]}")
