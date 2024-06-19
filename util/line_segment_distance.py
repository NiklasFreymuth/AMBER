import numpy as np


def get_line_segment_distances(
        points: np.array,
        projection_segments: np.array,
        return_minimum: bool = False,
        return_tangent_points: bool = False,
        min_axis: int = 1,
) -> np.array:
    """
    Calculates the distances of an array of points to an array of line segments.
    Vectorized for any number of points and line segments
    Args:
        points: An array of shape [num_points, 2], i.e., an array of points to project towards the projection segments
        projection_segments: An array of shape [num_segments, 4], i.e., an array of line segments/point pairs.
            Each segment is defined as [x1, y1, x2, y2], where (x1, y1) and (x2, y2) are the start and end points of the
            line segment, respectively.
        return_minimum: If True, the minimum distance is returned. If False, an array of all distances is returned
        return_tangent_points: If True, distances and tangent points of the projections to all segments are returned

    Returns: An array of shape [num_points, {num_segments, 1}] containing the distance of each point to each segment
        or the minimum segment, depending on return_minimum

    """
    segment_distances = projection_segments[:, :2] - projection_segments[:, 2:]
    tangent_positions = np.sum(projection_segments[:, :2] * segment_distances, axis=1) - points @ segment_distances.T
    segment_lengths = np.linalg.norm(segment_distances, axis=1)

    # the normalized tangent position is in [0,1] if the projection to the line segment is directly possible
    normalized_tangent_positions = tangent_positions / segment_lengths ** 2

    # it gets clipped to [0,1] otherwise, i.e., clips projections to the boundary of the line segment.
    # this is necessary since line segments may describe an internal part of the mesh domain, meaning
    # that we always want the distance to the segment rather than the distance to the line it belongs to
    normalized_tangent_positions[normalized_tangent_positions > 1] = 1  # clip too big values
    normalized_tangent_positions[normalized_tangent_positions < 0] = 0  # clip too small values
    tangent_points = projection_segments[:, :2] - normalized_tangent_positions[..., None] * segment_distances
    projection_vectors = points[:, None, :] - tangent_points

    distances = np.linalg.norm(projection_vectors, axis=2)
    if return_minimum:
        distances = np.min(distances, axis=min_axis)
    if return_tangent_points:
        return distances, tangent_points
    return distances


def min_line_segment_distance(query_segment, reference_segments: np.ndarray, num_query_points: int = 21):
    """
    Calculate the minimum distance of a query segment to a set of reference segments
    Args:
        query_segment: A line segment of shape (4,) representing the query segment. The first two elements are the
            start point, the last two elements are the end point, i.e., the segment is (x0, y0, x1, y1)
        reference_segments: Reference segments of shape (-1, 4), where the last dimension is over
          (x0, y0, x1, y1) for each segment
        num_query_points: Currently do this sample-based for a number of points. This is the number of samples
        we take and take the minimum over

    Returns:

    """
    points_along_sample_segments = np.linspace(start=query_segment[:2],
                                               stop=query_segment[2:],
                                               num=num_query_points,
                                               endpoint=True,
                                               axis=0)
    return get_line_segment_distances(points_along_sample_segments, reference_segments,
                                      return_minimum=True, return_tangent_points=False, min_axis=0)
