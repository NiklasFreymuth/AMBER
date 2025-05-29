import copy
from typing import Tuple

import numpy as np
from omegaconf import DictConfig
from skfem import adaptive_theta

from src.algorithm.dataloader.source_data import SourceData
from src.helpers.qol import filter_included_fields
from src.tasks.dataset_preparator import DatasetPreparator
from src.tasks.domains import get_initial_mesh_from_domain_config
from src.tasks.domains.extended_mesh_tri1 import ExtendedMeshTri1
from src.tasks.domains.geometry_util import volume_to_edge_length
from src.tasks.domains.gmsh_util import generate_initial_mesh
from src.tasks.domains.mesh_wrapper import MeshWrapper
from src.tasks.features.fem.elliptic_problem.elliptic_problem import EllipticProblem
from src.tasks.features.fem.elliptic_problem.laplace_problem import LaplaceProblem
from src.tasks.features.fem.elliptic_problem.poisson_problem import PoissonProblem
from src.tasks.features.fem.fem_problem import FEMProblem


class EllipticDataGenerator(DatasetPreparator):
    def __init__(
        self,
        algorithm_config: DictConfig,
        task_config: DictConfig,
    ):
        super().__init__(algorithm_config=algorithm_config, task_config=task_config)
        self.heuristic_config = task_config.get("refinement_heuristic")
        self.domain_config = task_config.get("domain")
        self.fem_config = task_config.get("fem")
        self.dataset_config = task_config.get("dataset")
        self.pde_features = filter_included_fields(self.fem_config.get("pde_features", {}))

        if self.fem_config.name == "poisson":
            self.problem_class = PoissonProblem
        elif self.fem_config.name == "laplace":
            self.problem_class = LaplaceProblem
        else:
            raise NotImplementedError(f"Problem class {self.fem_config.name=} not implemented")

    def _prepare_source_and_mesh(self, data_idx: int, dataset_mode: str) -> Tuple[SourceData, MeshWrapper]:
        """
        Generate a single data point
        Args:
        Returns:
        """
        # For training data, we want to have different random seeds for each data point. For validation and test data,
        # we want to have the same random seed for each data point, also across trials and runs.
        if dataset_mode == "train":
            seed = np.random.randint(0, 2**31)
        elif dataset_mode == "val":
            seed = (2**31) - data_idx
        else:
            seed = data_idx  # for test data, we want to have the same seed for each data point

        random_state = np.random.RandomState(seed=seed)
        max_initial_element_volume = self.max_initial_element_volume(bounding_box=np.array([0, 0, 1, 1]), dimension=2)
        initial_mesh = get_initial_mesh_from_domain_config(
            domain_config=self.domain_config,
            max_initial_element_volume=max_initial_element_volume,
            random_state=copy.deepcopy(random_state),
        )
        fem_problem = self.problem_class(
            fem_config=self.fem_config,
            initial_mesh=initial_mesh,
            observation_features=self.pde_features,
            random_state=random_state,
        )
        expert_mesh = self._get_heuristic_mesh(fem_problem, reference_mesh=initial_mesh)
        expert_mesh.geom_fn = initial_mesh.geom_fn
        expert_mesh = MeshWrapper(expert_mesh)
        initial_mesh = MeshWrapper(initial_mesh)
        source_data = SourceData(expert_mesh=expert_mesh, initial_mesh=initial_mesh, feature_provider=fem_problem)
        return source_data, initial_mesh

    def _get_heuristic_mesh(self, fem_problem: FEMProblem, reference_mesh: ExtendedMeshTri1) -> ExtendedMeshTri1:
        """
        Gets one heuristic expert mesh from the given fem_problem and reference (initial) mesh

        Args:
            fem_problem:
            reference_mesh:

        Returns:

        """
        assert isinstance(fem_problem, EllipticProblem)
        refinement_steps = self.heuristic_config.get("refinement_steps")
        error_threshold = self.heuristic_config.get("error_threshold")
        smooth_mesh = self.heuristic_config.get("smooth_mesh")
        initial_volume = self.heuristic_config.get("step0_element_volume")

        dim = reference_mesh.dim()
        target_edge_length = volume_to_edge_length(initial_volume, dim=dim)
        geom_fn = reference_mesh.geom_fn

        heuristic_mesh = generate_initial_mesh(geom_fn, target_edge_length, dim=dim, target_class=reference_mesh.__class__)
        for _ in range(refinement_steps):
            solution = fem_problem.calculate_solution(heuristic_mesh)
            error_indicator = fem_problem.get_error_indicator(mesh=heuristic_mesh, solution=solution)
            heuristic_mesh = heuristic_mesh.refined(adaptive_theta(error_indicator, error_threshold))
            if smooth_mesh:
                heuristic_mesh = heuristic_mesh.smoothed()
        return heuristic_mesh
