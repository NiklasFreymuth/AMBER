from typing import Optional, Tuple, Union

import torch
from torch_geometric.data import Batch, Data

from src.algorithm.loss.mesh_generation_loss import MeshGenerationLoss


class ImageAmberLoss(MeshGenerationLoss):
    def calculate_loss(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the loss for a batch of predictions and labels.
        Args:
            predictions: The predictions of the model
            labels: The labels for this batch of predictions. These may be per pixel for an image or
                per element/vertex of a mesh.

        Returns: A tuple consisting of the loss and the label-wise differences

        """
        labels = self.label_transform.inverse(labels, is_train=True)

        differences = self.get_differences(predictions, labels)
        element_loss = differences**2

        loss = torch.mean(element_loss)
        return loss, differences

    def get_differences(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate the differences between the predictions and the labels.
        Args:
            predictions: The predictions of the model
            labels: The labels for this batch of predictions. These may be per pixel for an image or
                per element/vertex of a mesh.

        Returns: The label-wise differences

        """
        differences = torch.abs(predictions - labels)
        return differences
