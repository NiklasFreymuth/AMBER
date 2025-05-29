import warnings
from functools import cached_property
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.algorithm.architecture import get_cnn
from src.algorithm.core.inference_step_output import InferenceStepOutput
from src.algorithm.core.mesh_generation_algorithm import MeshGenerationAlgorithm
from src.algorithm.dataloader.image_amber_data import ImageAmberData
from src.algorithm.loss.image_amber_loss import ImageAmberLoss
from src.algorithm.loss.mesh_generation_loss import MeshGenerationLoss
from src.algorithm.util.image_amber_util import get_feature_batch
from src.algorithm.visualization.amber_visualization import get_learner_plot
from src.algorithm.visualization.image_amber_visualization import (
    get_feature_image_plots,
    get_prediction_image_plots,
)
from src.helpers.custom_types import MeshGenerationStatus, MetricDict, PlotDict
from src.helpers.qol import prefix_keys
from src.helpers.torch_util import detach, make_batch
from src.mesh_util.mesh_metrics import MeshMetrics
from src.mesh_util.sizing_field_util import (
    scale_sizing_field_to_budget,
    sizing_field_to_num_elements,
)
from src.mesh_util.transforms.mesh_to_image import MeshImage
from src.tasks.domains.mesh_wrapper import MeshWrapper


class ImageAmber(MeshGenerationAlgorithm):
    """
    Like Amber, but on images. Uses a single image as input, and predicts a sizing field per pixel.
    No iterative rollout, no online data generation.
    Uses a CNN to predict the sizing field, so also no GNN.
    """

    ###################
    # Algorithm setup #
    ###################

    def _get_optimization_criterion(self) -> MeshGenerationLoss:
        criterion = ImageAmberLoss(label_transform=self.prediction_transform)
        return criterion

    def _get_model(self) -> nn.Module:
        return get_cnn(architecture_config=self.config.architecture, example_mesh_image=self.train_dataset.first.observation)

    ##################
    # Start training #
    ##################
    def _log_constant_plots(self, dataloader: DataLoader, prefix: str) -> None:
        super()._log_constant_plots(dataloader, prefix=prefix)
        for plotting_sample_idx in self.plotting_sample_idxs:
            if len(dataloader) <= plotting_sample_idx:
                continue
            data = dataloader.dataset[plotting_sample_idx]
            mesh_image = data.observation
            feature_plots = get_feature_image_plots(feature_names=self.in_feature_names, mesh_image=mesh_image)
            feature_plots = prefix_keys(feature_plots, prefix=f"features_{prefix}{plotting_sample_idx}")

            self._log_plots(feature_plots)

    #################
    # Training loop #
    #################

    def _training_step(self, batch: List[MeshImage], batch_idx: int) -> Tuple[torch.Tensor, MetricDict]:
        feature_batch, old_shapes = get_feature_batch(batch=batch, device=self.device)
        predictions = self._predict(feature_batch, is_train=True, flatten=False)

        # retrieve the original image shapes for each entry of the batch.
        # Within this old shape, find the valid pixels that correspond to a mesh, as only those matter for the loss
        in_mesh_predictions = []
        for prediction, mesh_image, old_shape in zip(predictions, batch, old_shapes):
            prediction = prediction.reshape(*feature_batch.shape[2:])
            slices = [slice(0, s) for s in old_shape]
            in_mesh_prediction = prediction[slices].flatten()[mesh_image.is_mesh]
            in_mesh_predictions.append(in_mesh_prediction)
        in_mesh_predictions = torch.cat(in_mesh_predictions)

        labels = torch.cat([mesh_image.labels for mesh_image in batch])  # big flat 1d tensor of valid labels
        labels = labels.to(self.device)
        loss, differences = self.criterion.calculate_loss(predictions=in_mesh_predictions, labels=labels)

        # log metrics and training scalars
        with torch.no_grad():
            batch_scalars = {
                "loss": loss.item(),
                "min_dif": torch.min(differences).item(),
                "max_dif": torch.max(differences).item(),
                "mean_dif": torch.mean(differences).item(),
            }
        return loss, batch_scalars

    ##########################
    # Evaluation and testing #
    ##########################
    def _evaluate_data_point(self, data: ImageAmberData) -> MetricDict:
        """
        Evaluation of a single data point (geometry -> series of meshes) during the evaluation phase of the algorithm.
        Args:
            data:

        Returns:

        """
        inference_output = self._inference_step(data)
        new_mesh = inference_output.output_mesh
        predictions = inference_output.predictions

        mesh_image = data.observation

        # get "online" metrics about the prediction, independent of whether its successful
        differences = self.criterion.get_differences(predictions=predictions, labels=mesh_image.labels.to(self.device))
        online_metrics = {
            "sf_MAE": torch.mean(differences).item(),
            "sf_MSE": torch.mean(differences**2).item(),
            "refinement_success": int(inference_output.refinement_success),
            "refinement_scaled": int(inference_output.refinement_scaled),
            "refinement_okay": int(inference_output.refinement_okay),
            "cur_elements": new_mesh.nelements,
            "cur_element_ratio": new_mesh.nelements / data.expert_mesh.nelements,
            "min_sf": np.min(self._clip_detach(predictions)),
        }

        # for the one-step baselines, we evaluate mesh metrics regardless of if the refinement works or not, to also
        # capture metrics for failed samples. We do the same for AMBER, where we always evaluate the last successful
        # step. In both cases, we expect success rates above 99% for converged trainings, so catching these one-off
        # failures in this way should not affect overall results too much.
        mesh_metrics = MeshMetrics(
            metric_config=self.config.mesh_metrics, reference_mesh=data.expert_mesh, evaluated_mesh=new_mesh, fem_problem=data.fem_problem
        )()
        return online_metrics | mesh_metrics

    def _visualize_data_point(self, data: ImageAmberData) -> PlotDict:
        """
        May provide arbitrary functions here that are used to draw additional plots.
        Args:
        Returns: A dictionary of {plot_name: plot}, where plot_function is any function that takes
          this algorithm at a current point as an argument, and returns a plotly figure.

        """
        mesh_image: MeshImage = data.observation

        inference_output = self._inference_step(data)
        new_mesh = inference_output.output_mesh
        predictions: np.ndarray = detach(inference_output.predictions)

        # We only predict "valid" pixels, i.e., those that correspond to a mesh element
        # To properly visualize our image, we now need to back-fill the predictions into the grid
        is_mesh = mesh_image.is_mesh
        label_grid = detach(mesh_image.label_grid)
        prediction_grid = np.zeros(np.prod(label_grid.shape))
        prediction_grid[is_mesh] = predictions
        prediction_grid = prediction_grid.reshape(label_grid.shape)

        plots = get_prediction_image_plots(mesh_image=mesh_image, prediction_grid=prediction_grid)

        fem_problem = data.fem_problem
        if fem_problem is not None:
            solution = fem_problem.calculate_solution(new_mesh)
        else:
            solution = None
        plots["mesh"] = get_learner_plot(predicted_mesh=new_mesh, solution=solution, mesh_generation_status=inference_output.mesh_generation_status)
        return plots

    def _inference_step(self, data: ImageAmberData) -> InferenceStepOutput:
        """
        Perform a single step of iterative inference, i.e. predict a sizing field, refine the mesh, and return the new
        graph and mesh.
        Args:
            data: The data object containing the graph and mesh to refine

        Returns: A 3-tuple of
        * the predictions on the old graph,
        * the refined mesh, wrapped in a MeshWrapper,
        * refinement success, i.e., whether the refinement was skipped due to the max_mesh_elements constraint.
            If so, the refined mesh and new graph are the same as the input mesh and graph, respectively.


        """
        # 1. Predict values, clip to generate sizing field
        mesh_image = data.observation
        mesh = data.mesh

        mesh_generation_status: MeshGenerationStatus = "success"

        image = mesh_image.features
        predictions = self._predict(make_batch(image), is_train=False)
        predictions = predictions[mesh_image.is_mesh]  # filter valid predictions
        sizing_field = self._clip_detach(predictions)

        # 2. Estimate number of elements in the new mesh, make sure it is not too large
        if self.max_mesh_elements is not None:
            approx_new_num_elements = sizing_field_to_num_elements(
                mesh=mesh, sizing_field=sizing_field, node_type=self.mesh_node_type, pixel_volume=mesh_image.pixel_volume
            )
            if approx_new_num_elements > self.max_mesh_elements:
                if self.force_mesh_generation:
                    sizing_field = scale_sizing_field_to_budget(
                        sizing_field=sizing_field,
                        mesh=mesh,
                        max_elements=self.max_mesh_elements,
                        node_type=self.mesh_node_type,
                        pixel_volume=mesh_image.pixel_volume,
                    )
                    mesh_generation_status = "scaled"
                else:
                    # no refinement if the new mesh would be too large
                    return InferenceStepOutput(predictions, mesh, mesh_generation_status="failed")

        # 3. Construct new mesh from sizing field
        try:
            from src.tasks.domains.update_mesh import update_mesh

            new_mesh = update_mesh(
                old_mesh=mesh.mesh,
                sizing_field=sizing_field,
                gmsh_kwargs=self.gmsh_kwargs,
                sizing_field_positions=mesh_image.active_pixel_positions.T,
            )
        except Exception as e:
            warnings.warn(f"Mesh generation for sizing field of range {sizing_field.min()}, {sizing_field.max()} " f"failed with error: {e}")
            return InferenceStepOutput(predictions, mesh, mesh_generation_status="failed")

        return InferenceStepOutput(predictions, new_mesh, mesh_generation_status=mesh_generation_status)

    @cached_property
    def in_feature_names(self) -> List[str]:
        """
        List of all features going into the mesh. Used for visualization.
        Returns:

        """
        first_data_point = self.train_dataset.first
        feature_names: list = first_data_point.node_feature_names
        if first_data_point.feature_provider is not None:
            # ImageAmber always acts on elements, so take element features
            assert self.mesh_node_type == "pixel", "Must provide pixel-based estimate for ImageAmber"
            fem_feature_names = first_data_point.feature_provider.element_feature_names
            if fem_feature_names is not None:
                feature_names += fem_feature_names
        feature_names += ["is_mesh"]
        return feature_names
