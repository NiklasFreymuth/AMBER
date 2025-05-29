import copy
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from lightning import LightningModule
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from src.algorithm.dataloader.mesh_generation_data import MeshGenerationData
from src.algorithm.dataloader.mesh_generation_dataset import MeshGenerationDataset
from src.algorithm.normalizer import RunningNormalizer, get_normalizer
from src.algorithm.optimizer import get_optimizer_and_scheduler
from src.algorithm.prediction_transform import get_transform
from src.algorithm.util.amber_util import get_reconstructed_mesh
from src.algorithm.util.parse_input_types import get_mesh_node_type
from src.algorithm.visualization.amber_visualization import get_reference_plot
from src.helpers.qol import add_to_dictionary, aggregate_metrics, prefix_keys, safe_mean
from src.helpers.torch_util import count_parameters, detach
from src.mesh_util.mesh_metrics import MeshMetrics


class MeshGenerationAlgorithm(LightningModule, ABC):
    """
    Abstract class for the full mesh generation algorithm. This class provides a structured approach
    to training and evaluating a deep learning model for mesh generation.
    """

    def __init__(self, algorithm_config: DictConfig, train_dataset: MeshGenerationDataset):
        """
        Initialize the mesh generation algorithm with the given configuration and training dataset.

        Args:
            algorithm_config (DictConfig): Configuration parameters for the algorithm.
            train_dataset (MeshGenerationDataset): Dataset used for training the model.
        """
        super().__init__()

        self.config = algorithm_config
        self.train_dataset = train_dataset

        ##########################
        # Instantiate parameters #
        ##########################
        # How to derive the sizing field and what to do with it
        self.mesh_node_type = get_mesh_node_type(self.config.sizing_field_interpolation_type)

        # Mesh generation parameters
        self.force_mesh_generation = self.config.force_mesh_generation
        self.gmsh_kwargs: Dict[str, float] = self._get_gmsh_kwargs(gmsh_config=self.config.gmsh)
        self.max_mesh_elements: float = self._get_max_mesh_elements(self.config.max_mesh_elements)

        # Evaluation frequency
        self.evaluation_frequency = self.config.evaluation_frequency

        # Plotting configuration
        self.plotting_sample_idxs: List[int] = self.config.plotting.sample_idxs
        self.plot_frequency: int = self.config.plotting.frequency
        self.plot_initial_epoch: bool = self.config.plotting.initial_epoch

        ########################
        # Model and normalizer #
        ########################
        self.model = self._get_model()
        self.prediction_transform = get_transform(transform_config=algorithm_config.prediction_transform)
        self.criterion = self._get_optimization_criterion()
        self.normalizer = self._get_normalizer()
        ###########
        # Logging #
        ###########

        # Create dicts to save metrics over steps
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.grad_norms = []

        # save config as hyperparameter for loading
        self.save_hyperparameters("algorithm_config")

    ###################
    # Algorithm setup #
    ###################

    def _get_model(self):
        """
        Abstract method to define and return the model.

        Returns:
            The deep learning model used for mesh generation.
        """
        raise NotImplementedError

    def _get_optimization_criterion(self):
        """
        Abstract method to define and return the loss function used for optimization.

        Returns:
            Loss function used for training the model.
        """
        raise NotImplementedError

    def _get_normalizer(self) -> RunningNormalizer:
        """
        Initializes the normalizer based on the given configuration and dataset.
        The normalizer acts on online data and does 2 things:
        * It normalizes the input features (which can be a graph or an image, depending on the underlying algorithm)
            to be in N(0,1) per features
        * It denormalizes the predictions to the original scale of the data, potentially taking care of any prediction
            transformations (e.g., exp or softplus) acting on the predictions. Thus, it allows the model to learn on
            a transformed target space that is similar to N(0,1), while still predicting the original target space.

        This function is run during both training and evaluation.
        However, we overwrite the initial normalizer statistics with saved checkpoints if running in evaluation mode.

        Returns:
            RunningNormalizer: The normalizer for input data.
        """
        normalizer = get_normalizer(
            normalizer_config=self.config.normalizer,
            example_input=self.train_dataset.first.observation,
            prediction_transform=self.prediction_transform,
        )
        [normalizer.update_normalizers(x.observation) for x in self.train_dataset.data]
        return normalizer

    def _get_gmsh_kwargs(self, gmsh_config: DictConfig) -> Dict:
        """
        Computes the Gmsh parameters for mesh generation based on the dataset properties.

        Args:
            gmsh_config (DictConfig): Configuration for Gmsh parameters.

        Returns:
            Dict[str, float]: Dictionary containing min and max sizing fields for mesh generation.
        """
        from src.mesh_util.sizing_field_util import get_sizing_field

        min_sizing_field = gmsh_config.get("min_sizing_field")
        if min_sizing_field.startswith("x"):
            factor = 1 / float(min_sizing_field[1:])
            min_sizing_field = factor * np.min([np.min(get_sizing_field(mesh)) for mesh in self.train_dataset.expert_meshes])
        max_sizing_field = gmsh_config.get("max_sizing_field")
        if max_sizing_field.startswith("x"):
            factor = float(max_sizing_field[1:])
            max_sizing_field = factor * np.max([np.max(get_sizing_field(mesh)) for mesh in self.train_dataset.expert_meshes])

        return {"min_sizing_field": min_sizing_field, "max_sizing_field": max_sizing_field}

    def _get_max_mesh_elements(self, max_elements: str | float) -> float:
        """
        Determines the maximum number of mesh elements allowed.

        Args:
            max_elements (Union[str, float]): The maximum number of elements, either as a fixed value or as a scaling factor.

        Returns:
            float: The computed maximum number of mesh elements.
        """
        if isinstance(max_elements, str):
            max_data_elements = max([mesh.nelements for mesh in self.train_dataset.expert_meshes])
            if max_elements == "auto":
                max_elements = int(max_data_elements * 1.5)
            elif max_elements.startswith("x"):
                # has form xF, where F is a float. E.g., x1.5 --> max_elements == max_data_elements*1.2
                factor = float(max_elements[1:])
                max_elements = int(max_data_elements * factor)
        return max_elements

    def configure_optimizers(self) -> Optimizer | Dict[str, Optimizer | Dict[str, LRScheduler | str]]:
        """
        Configures the optimizer and, if applicable, the learning rate scheduler.

        Returns:
            Union[Optimizer, Dict[str, Union[Optimizer, Dict[str, Union[LRScheduler, str]]]]]]:
                The optimizer with or without a scheduler.
        """

        return get_optimizer_and_scheduler(optimizer_dict=self.config.optimizer, model=self.model, num_epochs=self.trainer.max_epochs)

    ##################
    # Start training #
    ##################
    def on_train_start(self):
        """
        Calculate on-time metrics
        Returns:

        """
        validation_loader = self.trainer.val_dataloaders
        self._log_constant_metrics(validation_loader)
        self._log_constant_plots(dataloader=validation_loader, prefix="val")

    def _log_constant_metrics(self, validation_loader) -> None:
        train_expert_meshes = self.train_dataset.expert_meshes
        initial_meshes = [x.mesh for x in self.train_dataset.data]
        constant_metrics = {
            "min_sizing_field": self.gmsh_kwargs.get("min_sizing_field"),
            "max_sizing_field": self.gmsh_kwargs.get("max_sizing_field"),
            "max_mesh_elements": self.max_mesh_elements,
            "mean_expert_elements": np.mean([mesh.nelements for mesh in train_expert_meshes]),
            "max_expert_elements": np.max([mesh.nelements for mesh in train_expert_meshes]),
            "min_expert_elements": np.min([mesh.nelements for mesh in train_expert_meshes]),
            "mean_expert_vertices": np.mean([mesh.nvertices for mesh in train_expert_meshes]),
            "max_expert_vertices": np.max([mesh.nvertices for mesh in train_expert_meshes]),
            "min_expert_vertices": np.min([mesh.nvertices for mesh in train_expert_meshes]),
            "mean_initial_elements": np.mean([mesh.nelements for mesh in initial_meshes]),
            "max_initial_elements": np.max([mesh.nelements for mesh in initial_meshes]),
            "min_initial_elements": np.min([mesh.nelements for mesh in initial_meshes]),
            "mean_initial_vertices": np.mean([mesh.nvertices for mesh in initial_meshes]),
            "max_initial_vertices": np.max([mesh.nvertices for mesh in initial_meshes]),
            "min_initial_vertices": np.min([mesh.nvertices for mesh in initial_meshes]),
            "num_network_parameters": count_parameters(self.model),
        }
        metric_dict_list = {}
        for data in tqdm(validation_loader, desc="Initial Metrics".title()):
            data: MeshGenerationData
            sample_dict = self._evaluate_initial_sample(data=data)
            add_to_dictionary(metric_dict_list, new_scalars=sample_dict)
        metric_dict_list = {key: safe_mean(value) for key, value in metric_dict_list.items()}
        metric_dict_list = prefix_keys(metric_dict_list, prefix="constant", separator=".")
        constant_metrics = prefix_keys(constant_metrics, prefix="constant", separator="/")
        constant_metrics = constant_metrics | metric_dict_list

        self.log_dict(constant_metrics, on_step=False, prog_bar=True)

    def _evaluate_initial_sample(self, data: MeshGenerationData) -> Dict[str, float]:
        initial_mesh = data.mesh
        expert_mesh = data.expert_mesh
        # get comparison between expert and initial mesh.
        initial2expert_similarity_metrics = MeshMetrics(
            metric_config=self.config.mesh_metrics, reference_mesh=expert_mesh, evaluated_mesh=initial_mesh, fem_problem=data.fem_problem
        )()
        initial2expert_similarity_metrics = prefix_keys(initial2expert_similarity_metrics, prefix="initial")
        # get comparison between the expert and an "ideal" reconstruction of the expert's sizing field
        reconstructed_mesh = get_reconstructed_mesh(expert_mesh, gmsh_kwargs=self.gmsh_kwargs)
        reconstruction2expert_similarity_metrics = MeshMetrics(
            metric_config=self.config.mesh_metrics, reference_mesh=expert_mesh, evaluated_mesh=reconstructed_mesh, fem_problem=data.fem_problem
        )()
        reconstruction2expert_similarity_metrics = prefix_keys(reconstruction2expert_similarity_metrics, prefix="rec")
        sample_dict = reconstruction2expert_similarity_metrics | initial2expert_similarity_metrics
        return sample_dict

    def _log_constant_plots(self, dataloader: DataLoader, prefix: str) -> None:
        for plotting_sample_idx in self.plotting_sample_idxs:
            if len(dataloader.dataset) <= plotting_sample_idx:
                continue
            data = dataloader.dataset[plotting_sample_idx]
            expert_plot = get_reference_plot(
                reference_mesh=data.expert_mesh,
                fem_problem=copy.deepcopy(data.fem_problem),
                reference_name="Expert",
            )
            self._log_plots({f"expert/{prefix}{plotting_sample_idx}": expert_plot})

    #################
    # Training loop #
    #################
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, scalars = self._training_step(batch, batch_idx)
        self.training_step_outputs.append(scalars)
        return loss

    @abstractmethod
    def _training_step(self, batch, batch_idx) -> torch.Tensor:
        raise NotImplementedError

    def on_after_backward(self):
        total_norm = detach(torch.norm(torch.stack([p.grad.norm() for p in self.parameters() if p.grad is not None]), p=2))
        self.grad_norms.append(total_norm)

    def on_train_epoch_end(self) -> None:
        # Log mean of training metrics over epoch to WandB
        epoch_averages = aggregate_metrics(metrics=self.training_step_outputs)
        epoch_averages = prefix_keys(epoch_averages, prefix="metrics.train")
        epoch_averages["grad_norm"] = np.mean(self.grad_norms)
        self.log_dict(epoch_averages, on_epoch=True, prog_bar=True)
        self.training_step_outputs.clear()  # free memory

    ##########################
    # Evaluation and testing #
    ##########################
    def validation_step(self, batch, batch_idx: int) -> None:
        if self.current_epoch % self.evaluation_frequency == 0:
            evaluation_dict = self._evaluate_data_point(data=batch)
            self.validation_step_outputs.append(evaluation_dict)

        if self.current_epoch % self.plot_frequency == 0 and (self.plot_initial_epoch or self.current_epoch > 0):
            if batch_idx in self.plotting_sample_idxs:
                # Plot some validation samples
                plot_dict = self._visualize_data_point(data=batch)
                plot_dict = prefix_keys(plot_dict, prefix=f"val{batch_idx}")
                self._log_plots(plot_dict)

    def test_step(self, batch, batch_idx: int) -> None:
        evaluation_dict = self._evaluate_data_point(data=batch)
        self.test_step_outputs.append(evaluation_dict)

        if batch_idx in self.plotting_sample_idxs:
            # Plot some validation samples
            plot_dict = self._visualize_data_point(data=batch)
            plot_dict = prefix_keys(plot_dict, prefix=f"test{batch_idx}")
            self._log_plots(plot_dict)

    def _evaluate_data_point(self, data) -> Dict:
        raise NotImplementedError

    def _visualize_data_point(self, data) -> Dict:
        raise NotImplementedError

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) > 0:  # have some evaluation to log
            validation_averages = aggregate_metrics(metrics=self.validation_step_outputs)
            validation_averages = prefix_keys(validation_averages, prefix="metrics.val", separator="_")
            self.log_dict(validation_averages, on_epoch=True, prog_bar=True)
            self.validation_step_outputs.clear()  # free memory

    def on_test_end(self) -> None:
        test_averages = aggregate_metrics(metrics=self.test_step_outputs)
        test_averages = prefix_keys(test_averages, prefix="metrics.test", separator="_")
        self.logger.experiment.log(test_averages, step=self.current_epoch)

        # log metrics as dataframe/table
        test_step_outputs_df = pd.DataFrame(self.test_step_outputs)
        self.logger.experiment.log({"test_table": test_step_outputs_df}, step=self.current_epoch)
        self.test_step_outputs.clear()  # free memory

        # log expert meshes for test set
        test_loader = self.trainer.test_dataloaders
        self._log_constant_plots(dataloader=test_loader, prefix="test")

    ############################
    # prediction/model forward #
    ############################

    def _predict(self, batch: Batch | torch.Tensor, is_train: bool = False, flatten: bool = True) -> torch.Tensor:
        """
        Predict the output for a (batch of) graph(s).
        Args:
            batch: Batch of graphs or images to predict the output for
            is_train: Whether the model is currently in training mode and the prediction will be used to compute a loss

        Returns: A 1d tensor of predictions corresponding to the number of mesh elements in the input batch

        """
        batch = batch.to(self.device)
        batch = self.normalizer.normalize_inputs(batch)
        predictions = self.model(batch)
        if flatten:
            predictions = predictions.flatten()

        predictions = self.normalizer.denormalize_predictions(predictions)

        # Add baseline if exists to allow predicting residuals.
        if hasattr(batch, "current_sizing_field"):
            current_sizing_field = batch.current_sizing_field
        else:
            current_sizing_field = None
        predictions = self.prediction_transform.forward(predictions, baseline=current_sizing_field, is_train=is_train)
        return predictions

    def _clip_detach(self, sizing_field: torch.Tensor) -> np.ndarray:
        sizing_field = np.clip(detach(sizing_field), self.gmsh_kwargs.get("min_sizing_field"), self.gmsh_kwargs.get("max_sizing_field"))
        return sizing_field

    ###########
    # logging #
    ###########

    def _log_plots(self, plot_dict: Dict[str, go.Figure]):
        plot_dict = prefix_keys(plot_dict, prefix="figure", separator=".")
        # import wandb
        # wandb.log(plot_dict)
        plot_dict["epoch"] = self.current_epoch
        self.logger.experiment.log(plot_dict, step=int(plot_dict["epoch"]))
