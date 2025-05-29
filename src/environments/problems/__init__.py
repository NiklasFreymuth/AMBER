from typing import Any, Dict, Union, List

import numpy as np
from skfem import Mesh

from src.environments.problems.problems_2d.abstract_finite_element_problem_2d import (
    AbstractFiniteElementProblem,
)


def create_finite_element_problem(
        *,
        fem_config: Dict[Union[str, int], Any],
        initial_mesh: Mesh,
        element_features: List[str],
        random_state: np.random.RandomState,

) -> AbstractFiniteElementProblem:
    """
    Builds and returns a finite element problem class.
    Args:
        fem_config: Config containing additional details about the finite element method.
        initial_mesh: The initial mesh to use for the problem. Also stores the geometry of the problem
        element_features: Features to extract from the PDE
        random_state: The RandomState to use to draw functions in the __init__() and reset() calls of the target
            function class

    Returns: Some domain class that inherits from AbstractDomain.

    """
    pde_type = fem_config.get("pde_type")
    pde_type = pde_type.lower() if isinstance(pde_type, str) else None
    dimension = fem_config.get("domain", {}).get("dimension")
    if pde_type is None:
        assert dimension == 3
        from src.environments.problems.problems_3d.dummy_problem import (
            DummyProblem,
        )
        fem_problem = DummyProblem
    elif pde_type == "poisson":
        assert dimension == 2, "Only 2D Poisson problems are supported."
        from src.environments.problems.problems_2d.poisson import Poisson
        fem_problem = Poisson
    else:
        raise ValueError(f"Unknown pde_type: {pde_type}")
    return fem_problem(fem_config=fem_config,
                       initial_mesh=initial_mesh,
                       element_features=element_features,
                       random_state=random_state)
