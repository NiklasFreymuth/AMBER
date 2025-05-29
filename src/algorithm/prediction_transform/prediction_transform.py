import abc

import torch
from omegaconf import DictConfig


class PredictionTransform(abc.ABC):
    """
    Abstract base class for applying transformations to predicted values in a learning setting.

    This class supports two learning approaches for predicting non-negative values (e.g., sizing fields on a mesh):

    1. **Learning in the inverse space:** The loss is applied to transformed values, such as log-values or other inverse transformations.
    2. **Learning in the regular space:** The network outputs predictions directly, and a transformation is applied post hoc.

    It additionally supports learning the residual of the transformed values, which is useful for predicting relative changes.
    Here, we use the relation:

                x = f^-1(y)-f^-1(b)
            <-> y = f(x+f^-1(b))

    for labels y, predictions x, and baseline b

    Attributes:
        transform_config (DictConfig): Configuration dictionary specifying transformation settings.
        predict_residual (bool): Determines whether to predict the residual of the transformed values.
        inverse_transform_in_loss (bool): Determines whether to apply inverse transformation to labels during training.
    """

    def __init__(self, transform_config: DictConfig):
        """
        Initializes the transformation settings.

        Args:
            transform_config (DictConfig): Configuration object containing transformation settings.
        """
        self.transform_config = transform_config
        self.predict_residual = transform_config.predict_residual
        self.inverse_transform_in_loss = transform_config.inverse_transform_in_loss

    def __call__(self, prediction: torch.Tensor, baseline: torch.Tensor, is_train: bool = False) -> torch.Tensor:
        """
        Applies the transformation to the network's predictions.

        During training, if inverse_transform_in_loss is enabled, no transformation is applied,
        as the loss is computed on the network's direct output.

        Args:
            prediction (torch.Tensor): The raw prediction tensor from the model.
            baseline (torch.Tensor): The baseline tensor to add to the prediction iff self.predict_residual is True.
            is_train (bool, optional): Flag indicating whether the call is during training. Defaults to False.

        Returns:
            torch.Tensor: Transformed predictions if applicable, otherwise the raw predictions.
        """
        return self.forward(prediction, baseline, is_train)

    def forward(self, prediction: torch.Tensor, baseline: torch.Tensor = None, is_train: bool = False) -> torch.Tensor:
        """
        Applies the transformation to the network's predictions.

        During training, if inverse_transform_in_loss is enabled, no transformation is applied,
        as the loss is computed on the network's direct output.

        Args:
            prediction (torch.Tensor): The raw prediction tensor from the model.
            baseline (torch.Tensor, optional): The baseline tensor to add to the prediction iff self.predict_residual is True.
            is_train (bool, optional): Flag indicating whether the call is during training. Defaults to False.

        Returns:
            torch.Tensor: Transformed predictions if applicable, otherwise the raw predictions.
        """
        # Always apply transform during inference/evaluation case (is_train=False).
        # Apply to loss calculation only if self.inverse_transform_in_loss=False
        if self.inverse_transform_in_loss and is_train:
            pass  # Use raw output during training loss, since we apply inverse transformation to the labels.
        else:
            if self.predict_residual:
                # e.g., prediction = exp(prediction+log(baseline))
                # This comes from the relation
                # x = f^-1(y)-f^-1(b)
                # <-> y = f(x+f^-1(b))
                # for labels y, predictions x, and baseline b
                assert baseline is not None, "Baseline must be provided for residual prediction."
                # prediction is only the (inversed transformed) residual, so add the baseline back
                prediction = prediction + self._inverse_transform(baseline)
            prediction = self._transform(prediction)
        return prediction

    def inverse(self, labels: torch.Tensor, baseline: torch.Tensor = None, is_train: bool = False) -> torch.Tensor:
        """
        Applies the inverse transformation, usually to labels instead of predictions.

        If inverse_transform_in_loss is False, this implies learning happens in regular space,
        meaning the labels should not be transformed.

        Args:
            labels (torch.Tensor): The predicted values to be transformed.
            baseline (torch.Tensor, optional): The baseline tensor to add to the prediction iff self.predict_residual is True.
            is_train (bool, optional): Indicates if the input is a label. Defaults to False.

        Returns:
            torch.Tensor: The inverse-transformed prediction.
        """
        # Always apply to normalization cases. Apply to loss calculation if self.inverse_transform_in_loss is True
        # The below code is short for:
        # if self.inverse_transform_in_loss:
        #     if is_train:  # apply to loss case
        #         labels = self._inverse_transform(labels)  # e.g., labels = log(labels)
        #     else:  # Normalization case. Which statistics to use for the neural network output
        #         labels = self._inverse_transform(labels)
        # else:
        #     if is_train:  # Do not apply to loss case
        #         pass
        #     else:  # Normalization case. Which statistics to use for the neural network output
        #         labels = self._inverse_transform(labels)
        if self.inverse_transform_in_loss or not is_train:
            labels = self._inverse_transform(labels)
            if self.predict_residual:
                assert baseline is not None, "Baseline must be provided for residual prediction."
                # for the normalization case, we want to learn the inverse transformed residual of the labels, so
                labels = labels - self._inverse_transform(baseline)
        else:
            pass

        return labels

    @abc.abstractmethod
    def _transform(self, prediction: torch.Tensor) -> torch.Tensor:
        """
        Applies the forward transformation to the prediction.
        Must be implemented by subclasses.

        Args:
            prediction (torch.Tensor): The tensor to transform.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _inverse_transform(self, prediction: torch.Tensor) -> torch.Tensor:
        """
        Applies the inverse transformation to the prediction.
        Must be implemented by subclasses.

        Args:
            prediction (torch.Tensor): The tensor to inverse-transform.

        Returns:
            torch.Tensor: The inverse-transformed tensor.
        """
        raise NotImplementedError
