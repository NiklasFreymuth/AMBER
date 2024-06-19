"""
Utility class to select an algorithm based on a given config file
"""
from src.algorithms.abstract_iterative_algorithm import AbstractIterativeAlgorithm
from util.function import get_from_nested_dict
from util.types import *


def create_algorithm(config: ConfigDict, seed: Optional[int] = None) -> AbstractIterativeAlgorithm:
    algorithm_name = get_from_nested_dict(config, list_of_keys=["algorithm", "name"], raise_error=True).lower()

    # supervised learning algorithms
    if algorithm_name == "amber":
        from src.algorithms.amber.amber import AMBER

        return AMBER(config=config, seed=seed)

    elif algorithm_name == "image_amber":
        from src.algorithms.baselines.image_amber import ImageAMBER
        return ImageAMBER(config=config, seed=seed)
    else:
        raise NotImplementedError(f"Algorithm {algorithm_name} not implemented.")
