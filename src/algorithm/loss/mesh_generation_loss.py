from src.algorithm.prediction_transform.prediction_transform import PredictionTransform


class MeshGenerationLoss:
    def __init__(self, label_transform: PredictionTransform):
        """

        Args:
            label_transform (PredictionTransform): A transform to map between network space and mesh space. May
                apply to labels to learn in the transformed space.
        """
        self.label_transform = label_transform

    def __call__(self, *args, **kwargs):
        return self.calculate_loss(*args, **kwargs)

    def calculate_loss(self, *args, **kwargs):
        raise NotImplementedError

    def get_differences(self, *args, **kwargs):
        raise NotImplementedError
