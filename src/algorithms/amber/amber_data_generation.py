import copy
from typing import List, Union

import numpy as np
from skfem import Mesh
from tqdm import tqdm

from src.algorithms.amber.mesh_wrapper import MeshWrapper
from src.algorithms.amber.dataloader.amber_dataloader import AMBERDataLoader
from src.algorithms.amber.dataloader.amber_online_dataloader import AMBEROnlineDataLoader
from src.algorithms.baselines.image_amber_dataloader import ImageAMBERDataLoader
from src.environments.domains.extended_mesh_tet1 import ExtendedMeshTet1
from src.environments.domains.extended_mesh_tri1 import ExtendedMeshTri1
from src.environments.problems import AbstractFiniteElementProblem
from util.function import filter_included_fields
from util.types import ConfigDict


def get_train_val_loaders(
        supervised_config: ConfigDict,
        task_config: ConfigDict,
        device: str,
        random_state: np.random.RandomState,
        algorithm_name: str = "amber"
) -> List[Union[AMBERDataLoader, AMBEROnlineDataLoader]]:
    """
    Wrapper function for generating expert data and creating train and validation data loaders
    Args:
        supervised_config:
        task_config:
        device:
        random_state: Numpy random state for generating random expert meshes, if an online heuristic expert is used.
    Returns:

    """

    input_kwargs = {
        "element_feature_names": filter_included_fields(task_config.get("element_features")),
        "device": device,
        "supervised_config": supervised_config,
    }

    if algorithm_name == "amber":
        input_kwargs["edge_feature_names"] = filter_included_fields(task_config.get("edge_features"))

    gmsh_kwargs = {"min_sizing_field": supervised_config.get("min_sizing_field")}

    fem_config = task_config.get("fem")
    expert_data_config = task_config.get("expert_data")
    max_initial_element_volume = expert_data_config.get("max_initial_element_volume")

    num_pde_dict = {
        "train": expert_data_config.get("num_train_pdes"),
        "val": expert_data_config.get("num_val_pdes"),
        "test": expert_data_config.get("num_test_pdes"),
    }

    seeds = {
        "train": random_state.randint(0, 2 ** 31),
        "val": 0,
        "test": 1,
    }

    mesh_dataset_name = expert_data_config.get("mesh_dataset_name")
    dataset_config = expert_data_config.get(mesh_dataset_name)

    buffers = []
    for data_type_key, num_pdes in num_pde_dict.items():
        seed = seeds[data_type_key]
        _random_state = np.random.RandomState(seed=seed)

        coarse_meshes = get_initial_meshes(fem_config,
                                           dataset_config=dataset_config,
                                           data_type_key=data_type_key,
                                           num_pdes=num_pdes,
                                           max_initial_element_volume=max_initial_element_volume,
                                           seed=seed
                                           )

        if algorithm_name == "amber":
            initial_meshes = coarse_meshes
            if data_type_key == "train":
                class_type = AMBEROnlineDataLoader
            else:
                class_type = AMBERDataLoader
        elif algorithm_name == "image_amber":
            target_resolution = supervised_config.get("image_resolution")
            initial_meshes = [refine_mesh(mesh, target_resolution=target_resolution)
                              for mesh in coarse_meshes]
            class_type = ImageAMBERDataLoader
        else:
            raise NotImplementedError(f"Algorithm {algorithm_name} not implemented")

        fem_problems = get_fem_problems(fem_config, initial_meshes=initial_meshes, seed=seed)
        if mesh_dataset_name == "heuristic":
            expert_meshes = generate_heuristic_expert_meshes(fem_problems,
                                                             coarse_meshes=coarse_meshes,
                                                             heuristic_config=dataset_config)
        elif mesh_dataset_name == "console":
            expert_meshes = load_meshes_from_folder(dataset_config=dataset_config,
                                                    data_type=data_type_key,
                                                    num_pdes=num_pdes)
        else:
            raise NotImplementedError(f"Expert type {expert_data_config.get('mesh_dataset_name')} not implemented")
        expert_meshes = [MeshWrapper(expert_mesh) for expert_mesh in expert_meshes]

        # determine minimum sizing field value from expert meshes
        if "min_sizing_field" in gmsh_kwargs and gmsh_kwargs["min_sizing_field"] == "auto" and data_type_key == "train":
            from src.algorithms.amber.amber_util import get_sizing_field
            min_expert_sizing_field = np.min([np.min(get_sizing_field(mesh)) for mesh in expert_meshes])
            gmsh_kwargs["min_sizing_field"] = min_expert_sizing_field

        data_loader = class_type(
            initial_meshes=initial_meshes,
            fem_problems=fem_problems,
            expert_meshes=expert_meshes,
            random_state=_random_state,
            gmsh_kwargs=gmsh_kwargs,
            **input_kwargs
        )
        buffers.append(data_loader)
    return buffers


def refine_mesh(mesh: Union[ExtendedMeshTri1, ExtendedMeshTet1], target_resolution: int):
    """
    Refines a mesh to the target resolution
    Args:
        mesh:
        target_resolution: Resolution to refine the mesh to. This is the approximate number of elements in any direction

    Returns:

    """
    from src.algorithms.amber.amber_util import volume_to_edge_length
    from src.environments.domains.gmsh_util import generate_initial_mesh

    dim = mesh.dim()
    geom_bb = mesh.geom_bounding_box
    geom_bb = geom_bb.reshape(2, -1)
    geom_volume = np.abs(np.prod(geom_bb[0] - geom_bb[1]))
    # simplex_volume = np.prod(range(1, dim + 1))
    max_coarse_element_volume = geom_volume / ((target_resolution ** dim) * 2)  # make elements a bit smaller
    target_edge_length = volume_to_edge_length(max_coarse_element_volume, dim=dim)

    geom_fn = mesh.geom_fn
    queried_mesh = generate_initial_mesh(geom_fn, target_edge_length, dim=mesh.dim(), target_class=mesh.__class__)
    return queried_mesh


def get_initial_meshes(fem_config: ConfigDict, dataset_config: ConfigDict,
                       data_type_key: str,
                       num_pdes: int,
                       max_initial_element_volume: float,
                       seed: int) -> List[Union[ExtendedMeshTri1, ExtendedMeshTet1]]:
    """
    Generates a list of initial meshes including geometry

    Args:
        fem_config: Configuration dictionary of the finite element problem.
        dataset_config: Configuration dictionary of the dataset.
        data_type_key: Key of the data type to generate meshes for.
        num_pdes: Number of finite element problems to generate.
        max_initial_element_volume: Maximum volume of the initial elements in the mesh.
        seed: Random seed for reproducibility.

    Returns:
        A list of Mesh instances representing the initialized finite element problems.
    """
    from src.environments.domains import get_initial_mesh
    random_state = np.random.RandomState(seed=seed)
    initial_meshes = []
    for idx in range(num_pdes):
        _random_state = np.random.RandomState(seed=random_state.randint(0, 2 ** 31))
        initial_mesh = get_initial_mesh(
            mesh_idx=idx,
            dataset_config=dataset_config,
            data_type_key=data_type_key,
            domain_config=fem_config.get("domain"),
            max_initial_element_volume=max_initial_element_volume,
            random_state=copy.deepcopy(_random_state),
        )
        initial_meshes.append(initial_mesh)
    return initial_meshes


def get_fem_problems(fem_config: ConfigDict,
                     initial_meshes: List[Mesh], seed: int) -> List[AbstractFiniteElementProblem]:
    """
    Generates a list of wrapped finite element problems.
    For each problem,
    * an initial mesh including a mesh geometry is generated.
    * pde-specific features, such as a load function or boundary conditions, are calculated.
    Args:
        fem_config: Config of the finite element problem class
        initial_meshes: The initial meshes for the problems
        seed: Random seed for reproducibility

    Returns: A list of wrapped finite element problems

    """
    from src.environments.problems import create_finite_element_problem

    random_state = np.random.RandomState(seed=seed)

    if fem_config.get("pde_type") is None:
        element_features = []
    else:
        from util.function import filter_included_fields
        pde_config = fem_config.get(fem_config.get("pde_type"))
        element_features = filter_included_fields(pde_config.get("element_features", {}))

    fem_problems = []
    for idx, initial_mesh in tqdm(enumerate(initial_meshes), desc="Generating FEM Problems"):
        _random_state = np.random.RandomState(seed=random_state.randint(0, 2 ** 31))
        fem_problem = create_finite_element_problem(
            fem_config=fem_config,
            initial_mesh=initial_mesh,
            element_features=element_features,
            random_state=copy.deepcopy(_random_state),
        )
        fem_problems.append(fem_problem)
    return fem_problems


def generate_heuristic_expert_meshes(fem_problems: List[AbstractFiniteElementProblem],
                                     coarse_meshes: List[Mesh],
                                     heuristic_config: ConfigDict) -> List:
    """
    Generates expert meshes using a heuristic approach for the Poisson problem.
    Args:
        fem_problems:
        coarse_meshes:
        heuristic_config:

    Returns:

    """
    from skfem import adaptive_theta
    from src.environments.problems.problems_2d.poisson import Poisson

    refinement_steps = heuristic_config.get("refinement_steps")
    error_threshold = heuristic_config.get("error_threshold")
    smooth_mesh = heuristic_config.get("smooth_mesh")

    expert_meshes = []
    for mesh, fem_problem in tqdm(zip(coarse_meshes, fem_problems), "Generating Expert Meshes"):
        assert isinstance(fem_problem, Poisson)
        for _ in range(refinement_steps):
            solution = fem_problem.calculate_solution(mesh)
            error_indicator = fem_problem.get_error_indicator(mesh=mesh, solution=solution)
            mesh = mesh.refined(adaptive_theta(error_indicator, error_threshold))
            if smooth_mesh:
                mesh = mesh.smoothed()
        expert_meshes.append(mesh)
    return expert_meshes


def load_meshes_from_folder(dataset_config: ConfigDict, data_type: str, num_pdes: int) -> List:
    """
    Loads all meshes in the given folder for the provided data_type
    Args:
        dataset_config: Config describing the dataset. Contains the folder path.
        data_type: Either "train"/"val"/"test"
        num_pdes: Number of problems to load

    Returns:

    """
    import os
    from util.keys import DATASET_ROOT_PATH

    folder_path = dataset_config.get("folder_path")
    folder_path: str = os.path.join(DATASET_ROOT_PATH, folder_path, data_type)

    meshes = []
    nas_files = sorted([file for file in os.listdir(folder_path) if file.endswith(".nas")])[:num_pdes]
    for file in nas_files:
        mesh_path = os.path.join(folder_path, file)
        mesh = _load_mesh_from_path(mesh_path)
        meshes.append(mesh)
    return meshes


def _load_mesh_from_path(mesh_path: str) -> Mesh:
    import meshio
    from skfem.io import from_meshio
    from skfem import MeshTet, MeshTri, Mesh
    msh = meshio.read(mesh_path)
    mesh: Mesh = from_meshio(msh)
    if isinstance(mesh, MeshTri):
        from src.environments.domains.extended_mesh_tri1 import \
            ExtendedMeshTri1
        mesh = ExtendedMeshTri1(mesh.p, mesh.t)
    elif isinstance(mesh, MeshTet):
        from src.environments.domains.extended_mesh_tet1 import \
            ExtendedMeshTet1
        mesh = ExtendedMeshTet1(mesh.p, mesh.t)
    else:
        raise ValueError(f"Unsupported mesh type {type(mesh)}")
    return mesh
