from typing import Optional, Tuple, Union

import torch
from torch_geometric.data import Batch, Data

from src.algorithm.loss.mesh_generation_loss import MeshGenerationLoss
from src.algorithm.prediction_transform.prediction_transform import PredictionTransform


class AmberLoss(MeshGenerationLoss):
    def __init__(self, label_transform: PredictionTransform, loss_type: str = "mse"):
        """
        Computes the loss for mesh sizing field prediction using the AMBER framework.

        This loss function supports residual prediction via inverse-softplus transformation of
        target labels and applies the softplus transformation to model outputs to ensure positivity.

        The label transform (typically inverse softplus) maps target values into the untransformed space,
        where Mean Squared Error (MSE) or Mean Absolute Error (MAE) is computed. Optionally, a baseline/residual
        discrete sizing field can be subtracted from the target, allowing residual prediction.

        Given model output x_j and transformed target y_j, the loss is:
            L = (1/|V|) * Σ_j (x_j - softplus⁻¹(y_j))²  for MSE
        The predicted sizing field is then recovered as:
            f̂(v_j) = softplus(x_j + softplus⁻¹(b_j))

        Args:
            label_transform: A transform applied to invert softplus on target labels.
            loss_type: Type of loss to compute ('mse' or 'mae').
        """
        super().__init__(label_transform=label_transform)
        self.loss_type = loss_type

    def calculate_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        graph_batch: Optional[Batch] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the loss for a batch of predictions and labels.
        Args:
            predictions: The predictions of the model
            labels: The labels for this batch of predictions. These may be per pixel for an image or
                per element/vertex of a mesh.
            graph_batch: The graph batch containing the mesh data. This is used to determine the baseline/residual for the
                inverse softplus transformation. If the graph batch is None, the baseline is set to None.

        Returns: A tuple consisting of the loss and the label-wise differences

        """
        if graph_batch is not None and hasattr(graph_batch, "current_sizing_field"):
            baseline = graph_batch.current_sizing_field
        else:
            baseline = None
        labels = self.label_transform.inverse(labels, baseline=baseline, is_train=True)

        differences = self.get_differences(predictions=predictions, labels=labels)
        if self.loss_type == "mse":
            element_loss = differences**2
        elif self.loss_type == "mae":
            element_loss = differences  # l1 loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        loss = torch.mean(element_loss)
        return loss, differences

    def get_differences(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculates the (absolute) differences between the predictions and the labels as a torch tensor.
        Depending on the scope of the labels, these differences may require a mapping from the predictions to the labels,
            or a weighting of the predictions depending on their responsibilities w.r.t. the respective meshes they
            represent.
        Args:
            predictions:
            labels:

        Returns:

        """
        differences = torch.abs(predictions - labels)
        return differences
