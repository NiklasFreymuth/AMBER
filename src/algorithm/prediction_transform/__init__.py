from omegaconf import DictConfig

from src.algorithm.prediction_transform.prediction_transform import PredictionTransform


def get_transform(transform_config: DictConfig) -> PredictionTransform:
    transform_name = transform_config.name
    if transform_name in ["exp", "exponential"]:
        from src.algorithm.prediction_transform.exponential_transform import (
            ExponentialTransform,
        )

        return ExponentialTransform(transform_config)
    elif transform_name == "softplus":
        from src.algorithm.prediction_transform.softplus_transform import (
            SoftplusTransform,
        )

        return SoftplusTransform(transform_config)
    elif transform_name is None or transform_name in [False, "none", "null"]:
        from src.algorithm.prediction_transform.no_transform import NoTransform

        return NoTransform(transform_config)
    else:
        raise ValueError(f"Unsupported transform type: {transform_name=}")
