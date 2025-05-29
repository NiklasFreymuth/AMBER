from typing import Dict, List, Union, Optional

import numpy as np
from pykdtree.kdtree import KDTree
from skfem import Mesh

from src.algorithms.amber.mesh_wrapper import MeshWrapper
from util.function import prefix_keys


def get_similarity_metrics(refined_mesh: MeshWrapper, expert_mesh: Optional[MeshWrapper] = None,
                           reconstructed_mesh: Optional[MeshWrapper] = None) -> Dict[str, float]:
    """
    Calculate mesh similarity metrics between the expert mesh, the reconstructed mesh, and the refined mesh.
    Uses CachedMeshWrapper objects to cache expensive operations on the meshes, such as calculating element midpoints
    or simplex volumes.

    Args:
        refined_mesh:
        expert_mesh:
        reconstructed_mesh:

    Returns: A dictionary of mesh similarity metrics as key-value pairs

    """
    mesh_similarity_metrics = {}

    if expert_mesh is not None:
        mesh2expert_similarity_metrics = get_mesh_similarity_metrics(refined_mesh,
                                                                     mesh2=expert_mesh)
        mesh2expert_similarity_metrics = prefix_keys(mesh2expert_similarity_metrics, prefix="exp")
        mesh_similarity_metrics |= mesh2expert_similarity_metrics

    if reconstructed_mesh is not None:
        mesh2reconstruction_similarities = get_mesh_similarity_metrics(refined_mesh,
                                                                       mesh2=reconstructed_mesh)
        mesh2reconstruction_similarities = prefix_keys(mesh2reconstruction_similarities, prefix="rec")
        mesh_similarity_metrics |= mesh2reconstruction_similarities

    if expert_mesh is not None and reconstructed_mesh is not None:
        expert2reconstruction_similarities = get_mesh_similarity_metrics(expert_mesh,
                                                                         mesh2=reconstructed_mesh)
        expert2reconstruction_similarities = prefix_keys(expert2reconstruction_similarities, prefix="exp_rec")
        mesh_similarity_metrics |= expert2reconstruction_similarities
    return mesh_similarity_metrics


def get_mesh_similarity_metrics(mesh1: MeshWrapper,
                                mesh2: MeshWrapper) -> Dict[str, float]:
    """
    Calculate similarity metrics between two meshes. These metrics are symmetric, and thus do not care about the mesh
    order.
    These metrics are used to evaluate the quality of the evaluated mesh w.r.t. the reference mesh
    The metrics include
    - the Chamfer distance between the evaluated mesh and the reference mesh, including a density-aware variance
    - the symmetric element size difference between the evaluated mesh and the reference mesh

    Args:
        mesh1: The first mesh to compare
        mesh2: The second mesh to compare

    Returns:

    """
    from util.function import prefix_keys

    evaluated_midpoints = mesh1.get_midpoints()
    reference_midpoints = mesh2.get_midpoints()

    distance_types = ["vanilla", "density_aware"]

    metric_dict = {
        "element_delta": mesh2.nelements - mesh1.nelements,
    }

    chamfer_distance_midpoints = chamfer_distance(set1=evaluated_midpoints,
                                                  set2=reference_midpoints,
                                                  tree1=mesh1.get_midpoint_tree(),
                                                  tree2=mesh2.get_midpoint_tree(),
                                                  distance_types=distance_types,
                                                  )
    volume_differences = symmetric_volume_differences(mesh1, mesh2)

    chamfer_distance_midpoints = prefix_keys(chamfer_distance_midpoints, "cd_midpoints", separator="_")
    volume_differences = prefix_keys(volume_differences, "vol_dif", separator="_")
    metric_dict = metric_dict | chamfer_distance_midpoints | volume_differences

    return metric_dict


def chamfer_distance(
        set1: np.ndarray, set2: np.ndarray,
        tree1: Optional[KDTree] = None,
        tree2: Optional[KDTree] = None,
        distance_types: Union[str, List[str]] = "vanilla"
) -> Dict[str, np.ndarray]:
    """
    Calculate the Chamfer distance between two sets of points. The Chamfer distance is a measure of the similarity
    between two sets of points, quantifying how much one set differs from the other.
    It is computed by averaging the squared distances from each point in one set to its nearest neighbor
    in the other set and vice versa.

    Here, we consider different variants of the Chamfer distance. Mathematically, the variants are defined as:
    vanilla:
        C(A, B) = 0.5 * (1/|A| * sum_{a in A} min_{b in B} ||a - b||^2 + 1/|B| * sum_{b in B} min_{a in A} ||a - b||^2)
    density_aware:
        This variant introduces exponentiated scaling, i.e., 1-exp(-||a - b||^2) instead of ||a - b||^2,
        and additionally accounts for the density of the points by
        weighting the distances with the inverse of the count of nearest neighbor assignments. I.e., it scales each
        distance as 1-(1/n_b)exp(-||a - b||^2), where n_b is the number of points in A that are assigned to b.

    Args:
        set1: The first set of points. numpy.ndarray of shape (N, D) containing N D-dimensional points.
        set2: The second set of points. numpy.ndarray of shape (M, D) containing M D-dimensional points.
        tree1: A KDTree object for set1. If None, a new tree will be created.
        tree2: A KDTree object for set2. If None, a new tree will be created.
        distance_types: The types of Chamfer distance to calculate.
            Can be a single type "vanilla", "density_aware", or "exponentiated",
            or a list containing any combination of these.

    Returns:
        A dictionary {distance_type: distance_value} for each requested Chamfer distance type.
    """
    if isinstance(distance_types, str):
        distance_types = [distance_types]
    if tree1 is None:
        tree1 = KDTree(set1)
    if tree2 is None:
        tree2 = KDTree(set2)

    distances1, indices1 = tree1.query(set2, k=1)
    distances2, indices2 = tree2.query(set1, k=1)

    distances = {}
    if "vanilla" in distance_types:
        distances["vanilla"] = 0.5 * (np.mean(distances1) + np.mean(distances2))
    if "density_aware" in distance_types:
        exp_distances1 = np.exp(-distances1)
        exp_distances2 = np.exp(-distances2)
        # scale each distance by the number of indices that it matches
        weighted_distances1 = exp_distances1 / np.bincount(indices1).astype(np.float32)[indices1]
        weighted_distances2 = exp_distances2 / np.bincount(indices2).astype(np.float32)[indices2]
        distances["density_aware"] = 0.5 * (np.mean(1 - weighted_distances1) + np.mean(1 - weighted_distances2))
    return distances


def symmetric_volume_differences(mesh1: MeshWrapper, mesh2: MeshWrapper) -> Dict[str, np.ndarray]:
    """
    Calculate the symmetric element size difference between two meshes.
    The symmetric element size difference is a measure of the similarity between two meshes.
    It is calculated as the mean of the relative differences of the element sizes between the two meshes.
    Mathematically, it is defined as the difference in size between each element in one mesh and
    its corresponding element in the other mesh, for both meshes.

    Args:
        mesh1: The first mesh
        mesh2: The second mesh, provided as an EvaluationMeshWrapper

    Returns: The symmetric element size difference between the two meshes

    """
    volumes1 = mesh1.get_simplex_volumes()
    volumes2 = mesh2.get_simplex_volumes()

    # calculate the difference in size between each element in mesh1, and its corresponding element in mesh2
    element_correspondences1 = _get_element_correspondences(mesh1, mesh2)
    differences1 = volumes2 - volumes1[element_correspondences1]

    # calculate the difference in size between each element in mesh2, and its corresponding element in mesh1
    element_correspondences2 = _get_element_correspondences(mesh2, mesh1)
    differences2 = volumes1 - volumes2[element_correspondences2]

    differences = np.abs(np.concatenate([differences1, differences2]))

    # calculate the symmetric element size difference
    return {
        "summed": np.sum(differences),
        "squared": np.sum(differences ** 2),
    }


def _get_element_correspondences(mesh1: Union[Mesh, MeshWrapper],
                                 mesh2: Union[Mesh, MeshWrapper]) -> np.ndarray:
    """
    Get the element correspondences between two meshes. This is an array of indices with mesh2.nelements elements in
    range(mesh1.nelements) that maps each element in mesh2 to its corresponding element in mesh1.
    Args:
        mesh1: The "base mesh" to map the other mesh to
        mesh2: The mesh to map to the base mesh.

    Returns: An array of indices with mesh2.nelements elements in range(mesh1.nelements)
        that maps each element in mesh2 to its corresponding element in mesh1.

    """
    mesh2_midpoints = mesh2.p[:, mesh2.t].mean(axis=1)
    element_finder = mesh1.element_finder()
    corresponding_elements = element_finder(*mesh2_midpoints)
    return corresponding_elements
