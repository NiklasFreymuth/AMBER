from typing import Any, Dict, Union

import numpy as np

from src.environments.problems.load_functions.abstract_target_function import (
    AbstractTargetFunction,
)


def create_load_function(
    *,
    target_function_config: Dict[Union[str, int], Any],
    boundary: np.array,
    fixed_target: bool,
    random_state: np.random.RandomState,
    dimension: int = 2,
) -> AbstractTargetFunction:
    """
    Builds and returns a density class.
    Args:
        target_function_config: Config containing additional details about the target function. Depends on the
            target function
        boundary: 2d-rectangle that defines the boundary that this function should act in
        fixed_target: Whether to use a fixed target function. If True, the same target function will be used
            throughout. If False, a family of target functions will be created and a new target function will be drawn
            from this family whenever the reset() method is called
        random_state: The RandomState to use to draw functions in the __init__() and reset() calls of the target
            function class
        dimension: The dimension of the problem. Defaults to 2.

    Returns: Some density/target function that inherits from AbstractTargetFunction.

    """
    from src.environments.problems.load_functions.gmm_density import GMMDensity
    target_function_instance = GMMDensity(
        target_function_config=target_function_config,
        boundary=boundary,
        fixed_target=fixed_target,
        random_state=random_state,
        dimension=dimension,
    )
    return target_function_instance
