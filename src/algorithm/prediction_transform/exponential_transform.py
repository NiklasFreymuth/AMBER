import torch

from src.algorithm.prediction_transform import PredictionTransform


class ExponentialTransform(PredictionTransform):
    """
    Exponential transformation: learns the log of the sizing field and applies the loss on regular values.

    The network outputs log-values, and the final prediction is obtained by exponentiation.

    Forward transformation:
        prediction' = exp(prediction)

    Inverse transformation:
        prediction' = log(prediction)
    """

    def _transform(self, prediction: torch.Tensor) -> torch.Tensor:
        """
        Applies the exponential transformation.

        Args:
            prediction (torch.Tensor): The raw prediction tensor.

        Returns:
            torch.Tensor: The exponentiated prediction tensor.
        """
        prediction = torch.exp(prediction)
        if torch.any(torch.isnan(prediction)):
            import warnings

            warnings.warn("Predictions contain NaN values.")
            prediction[torch.isnan(prediction)] = 100  # Replace NaNs with a default value
        return prediction

    def _inverse_transform(self, prediction: torch.Tensor) -> torch.Tensor:
        """
        Applies the logarithmic inverse transformation.

        Args:
            prediction (torch.Tensor): The transformed prediction tensor.

        Returns:
            torch.Tensor: The log-transformed prediction tensor.
        """
        return torch.log(prediction)
