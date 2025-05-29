from functools import lru_cache
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from tqdm import tqdm

from src.algorithms.abstract_iterative_algorithm import AbstractIterativeAlgorithm, filter_scalars
from src.algorithms.amber.amber_data_generation import get_train_val_loaders
from src.algorithms.amber.amber_metrics import get_similarity_metrics
from src.algorithms.amber.amber_util import calculate_loss
from src.algorithms.amber.amber_visualization import get_learner_plots, get_reference_plots
from src.algorithms.amber.mesh_wrapper import MeshWrapper
from src.algorithms.amber.dataloader.amber_data import MeshRefinementData
from src.algorithms.amber.dataloader.amber_online_dataloader import AMBEROnlineDataLoader
from src.algorithms.baselines.image_amber_dataloader import ImageAMBERDataLoader
from src.algorithms.baselines.image_amber_visualization import get_feature_image_plots, get_prediction_image_plots
from src.algorithms.baselines.mesh_to_image import MeshImage
from src.algorithms.baselines.unet import UNet
from src.algorithms.util.normalizers.abstract_running_normalizer import AbstractRunningNormalizer
from src.algorithms.util.normalizers.dummy_running_normalizer import DummyRunningNormalizer
from src.algorithms.util.normalizers.mesh_image_running_normalizer import MeshImageRunningNormalizer
from src.modules.abstract_architecture import AbstractArchitecture
from util.function import add_to_dictionary, prefix_keys, safe_mean
from util.torch_util.torch_util import detach, count_parameters
from util.types import *


class ImageAMBER(AbstractIterativeAlgorithm):
    def __init__(self, config: ConfigDict, seed: Optional[int] = None):
        """
        Initializes the Adaptive Meshing By Expert Reconstruction (AMBER) algorithm.
        """
        super().__init__(config=config)

        self.seed = seed

        supervised_config = self.algorithm_config.get("supervised")

        self.batches_per_iteration = supervised_config.get("batches_per_iteration")
        self._max_grad_norm: float = supervised_config.get("max_grad_norm")

        # generate the expert data and initial meshes on it
        data_loader_on_gpu = self.config["algorithm"]["data_loader_on_gpu"]
        data_loader_device = self.device if data_loader_on_gpu else torch.device("cpu")
        loader_random_state = np.random.RandomState(seed=self.seed)
        task_config = self.config.get("task")
        self._train_loader, self._validation_loader, self._test_loader = get_train_val_loaders(
            supervised_config=supervised_config,
            task_config=task_config,
            device=data_loader_device,
            random_state=loader_random_state,
            algorithm_name=self.algorithm_config.get("name")
        )
        self._train_loader: AMBEROnlineDataLoader

        # initialize the model and normalizer
        self._initialize_model_and_normalizer(supervised_config)

        if supervised_config.get("max_mesh_elements") == "auto":
            # make max size depend on largest mesh in the training set
            self.max_mesh_elements = int(max([mesh.nelements for mesh in self.train_loader.expert_meshes]) * 1.2)
        else:
            # either a fixed number or None
            self.max_mesh_elements = supervised_config.get("max_mesh_elements")

        # some additional parameters for the supervised learning and evaluation
        self.evaluation_idxs = supervised_config.get("evaluation_idxs")

        self.loss_type = supervised_config.get("loss_type").lower()

        self.gmsh_kwargs = self.train_loader.gmsh_kwargs

        self.transform_predictions = supervised_config.get("transform_predictions")

        self._initial_plots = True  # only plot the expert and reconstructed mesh once
        self._initial_metrics = True  # only log the initial metrics once

    def _initialize_model_and_normalizer(self, supervised_config):
        # optionally load the architecture (hmpn model, optimizers)
        # Also load the normalizer if one is used
        checkpoint_config = self.algorithm_config.get("checkpoint")
        if checkpoint_config is not None and checkpoint_config.get("experiment_name") is not None:
            checkpoint = self.load_from_checkpoint(checkpoint_config=checkpoint_config)
            assert isinstance(
                checkpoint.architecture, UNet
            ), f"checkpoint must contain a GraphDQNPolicy, given type: '{type(checkpoint.architecture)}'"
            self._model = checkpoint.architecture
            self._running_normalizer = checkpoint.normalizer
        else:
            self._model = self._build_model()
            self._running_normalizer = self._build_normalizer(normalizer_config=supervised_config.get("normalizer"))
            [self._running_normalizer.update_observation_normalizers(obs) for obs in self.train_loader.observations]

    def _build_model(self) -> UNet:
        assert self.network_config.get("type_of_base") == "unet", (f"Only UNet architectures supported for ImageAMBER, "
                                                                   f"given {self.network_config.get('type_of_base')}")

        return UNet(
            network_config=self.network_config,
            in_features=self.in_features,
            out_features=1,  # predict a scalar per pixel
            use_gpu=self.use_gpu,
            dim=self.image_dim,
        )

    def _build_normalizer(self, normalizer_config: ConfigDict) -> AbstractRunningNormalizer:
        """
        Build the normalizer for the observations and predictions of the model.
        Args:
            normalizer_config: Config describing what and how to normalize

        Returns: A normalizer object that can be used to normalize observations

        """
        normalize_observations = normalizer_config.get("normalize_observations")
        observation_clip = normalizer_config.get("observation_clip")
        if normalize_observations:
            return MeshImageRunningNormalizer(num_features=self.in_features,
                                              normalize_features=normalize_observations,
                                              observation_clip=observation_clip,
                                              device=self.device)
        else:
            return DummyRunningNormalizer()

    @property
    def in_features(self):
        return len(self.in_feature_names)

    @property
    def image_dim(self):
        return self.train_loader.first.mesh.mesh.dim()

    @property
    @lru_cache(maxsize=1)
    def in_feature_names(self) -> List[str]:
        feature_names: list = self.train_loader.element_feature_names
        if self.train_loader.fem_problems[0].element_feature_names is not None:
            feature_names += self.train_loader.fem_problems[0].element_feature_names
        feature_names += ["is_mesh"]
        return feature_names

    @property
    def train_loader(self) -> ImageAMBERDataLoader:
        return self._train_loader

    @property
    def validation_loader(self) -> ImageAMBERDataLoader:
        return self._validation_loader

    @property
    def test_loader(self) -> ImageAMBERDataLoader:
        return self._test_loader

    @property
    def model(self) -> AbstractArchitecture:
        return self._model

    @property
    def architecture(self) -> AbstractArchitecture:
        return self.model

    def fit_iteration(self) -> ValueDict:
        metric_dict_list = {}
        self.model.train()
        for batch_idx in tqdm(range(self.batches_per_iteration)):
            train_metrics = self._train_batch()
            train_metrics = prefix_keys(train_metrics, prefix="train")
            metric_dict_list = add_to_dictionary(metric_dict_list, new_scalars=train_metrics)
        mean_metrics = {key + "_mean": safe_mean(value) for key, value in metric_dict_list.items()}
        # last_metrics = {key + "_last": value[-1] for key, value in metric_dict_list.items()}
        metrics = mean_metrics  # | last_metrics
        return metrics

    def _train_batch(self) -> ValueDict:
        mesh_image_batch = self.train_loader.get_batch(batch_size=self.num_batch_images)
        feature_batch = torch.cat([mesh_image.features for mesh_image in mesh_image_batch], dim=0)
        feature_batch = feature_batch.to(self.device)

        predictions = self._predict(feature_batch, train=True)

        labels = torch.cat([mesh_image.labels for mesh_image in mesh_image_batch])  # big 1d tensor of valid labels
        labels = labels.to(self.device)

        in_mesh_predictions = torch.cat([prediction[mesh_image.is_mesh]
                                         for prediction, mesh_image in zip(predictions, mesh_image_batch)])

        differences = torch.abs(in_mesh_predictions - labels)

        loss = calculate_loss(predictions=in_mesh_predictions,
                              labels=labels,
                              loss_type=self.loss_type)

        self.model.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._max_grad_norm)  # Clip grad norm
        self.model.optimizer.step()
        # log metrics and training scalars

        with torch.no_grad():
            if self.loss_type == "log_mse":
                predictions = torch.exp(predictions)
                if torch.any(torch.isnan(predictions)):
                    import warnings
                    warnings.warn("Predictions contain NaN values. This may be due to a too large learning rate.")
                    predictions[torch.isnan(predictions)] = 100

            train_scalars = {
                "loss": loss.item(),
                "min_dif": torch.min(differences).item(),
                "max_dif": torch.max(differences).item(),
                "mean_dif": torch.mean(differences).item(),
                "learning_rate": self.model.optimizer.param_groups[0]["lr"],
            }

        if self.learning_rate_scheduler is not None:
            self.learning_rate_scheduler.step()
            train_scalars["learning_rate"] = self.learning_rate_scheduler.get_last_lr()[0]
        return train_scalars

    def _predict(self, batch: InputBatch, train: bool = False) -> torch.Tensor:
        """
        Predict the output for a (batch of) graph(s).
        Args:
            batch: Batch of graphs to predict the output for
            train: Whether the model is currently in training mode and the prediction will be used to compute a loss

        Returns: A 1d tensor of predictions corresponding to the number of mesh elements in the input batch

        """
        batch = self.running_normalizer.normalize_observations(batch)
        predictions = self.model(batch)
        predictions = predictions.reshape(predictions.shape[0], -1)  # flatten each prediction, keep batch dim

        # maybe transform predictions to a non-negative value
        if self.transform_predictions == "exp":
            # learn the log sizing field, compare loss on regular sizing field
            predictions = torch.exp(predictions)
        elif self.transform_predictions == "softplus":
            # take a softplus. This is potentially more stable than the exp for larger values
            predictions = torch.nn.functional.softplus(predictions)
        if self.loss_type == "log_mse" and not train:
            # train on the log sizing field, predict the actual sizing field
            predictions = torch.exp(predictions)
            if torch.any(torch.isnan(predictions)):
                import warnings
                warnings.warn("Predictions contain NaN values. This may be due to a too large learning rate.")
                predictions[torch.isnan(predictions)] = 100
        return predictions

    def evaluate(self) -> ValueDict:
        """
        Perform a full rollout for each sample of the evaluation buffer.
        Here, this rollout considers M steps of inference, where the model predicts a sizing field, refines the mesh,
        and generates a solution on this mesh, generates a graph from the mesh and solution, and predicts again.

        After each step, mesh and sizing field similarity metrics are calculated and logged.
        Args:
        Returns:

        """
        with torch.no_grad():
            evaluation_dict = self._evaluate_dataloader(dataloader=self.validation_loader, description="validation")

        if self._initial_metrics:
            self._initial_metrics = False
            initial_metrics = self._get_initial_metrics()
            initial_metrics = prefix_keys(initial_metrics, "initial")
            evaluation_dict = evaluation_dict | initial_metrics
        return evaluation_dict

    def _evaluate_dataloader(self, dataloader: ImageAMBERDataLoader, description: str) -> ValueDict:
        """
        Perform a full rollout over the provided dataloader/buffer.
        Here, this rollout considers M steps of inference, where the model predicts a sizing field, refines the mesh,
        and generates a solution on this mesh, generates a graph from the mesh and solution, and predicts again.

        After each step, mesh and sizing field similarity metrics are calculated and logged.
        Args:
            dataloader: The dataloader to iterate over. Will evaluate every sample in this loader
            description: Description string for prefix of the evaluation and the tqdm progress bar

        Returns: A dictionary of evaluation metrics for the dataloader

        """
        from util.torch_util.torch_util import detach

        self.model.train(False)

        metric_dict_list = {}
        for data_idx, data in tqdm(enumerate(dataloader), desc=description.title()):
            # gather a bunch of fem-specific data
            data: MeshRefinementData
            mesh_image: MeshImage = copy.deepcopy(data.observation)
            mesh: MeshWrapper = data.mesh

            predictions, refined_mesh, refinement_success = self._inference_step(mesh_image=mesh_image,
                                                                                 mesh=mesh,
                                                                                 )

            # online metrics
            differences = np.abs(detach(predictions) - detach(mesh_image.labels))
            online_metrics = {
                "sf_MAE": np.mean(differences),
                "sf_MSE": np.mean(differences ** 2),
                "refinement_success": int(refinement_success),
                "cur_elements": refined_mesh.nelements,
                "min_sf": np.min(self._clip_sizing_field(predictions)),
            }
            metric_dict_list = add_to_dictionary(metric_dict_list, new_scalars=online_metrics)

            if refinement_success:
                # calculate metrics for the refined mesh
                fem_idx: int = data.fem_idx
                expert_mesh = dataloader.expert_meshes[fem_idx]
                reconstructed_mesh = dataloader.get_reconstructed_mesh(fem_idx)

                mesh_similarity_metrics = get_similarity_metrics(refined_mesh, expert_mesh, reconstructed_mesh)
                metric_dict_list = add_to_dictionary(metric_dict_list, new_scalars=mesh_similarity_metrics)

            else:
                remaining_dict = {"refinement_success": 0}
                metric_dict_list = add_to_dictionary(metric_dict_list, new_scalars=remaining_dict)
        evaluation_dict = {key: safe_mean(value) for key, value in metric_dict_list.items()}
        evaluation_dict = prefix_keys(evaluation_dict, description)

        return evaluation_dict

    def _inference_step(
            self,
            mesh_image: MeshImage,
            mesh: MeshWrapper,
    ) -> Tuple[torch.Tensor, MeshWrapper, bool]:
        """
        Perform an inference step, i.e. predict a sizing field, refine the mesh, and return the new mesh.
        Args:
            mesh_image: The image to predict the sizing field for
            mesh: The scikit FEM mesh to refine

        Returns: A 3-tuple of
        * the predictions on the old graph,
        * the refined mesh, wrapped in a CachedMeshWrapper,
        * refinement success, i.e., whether the refinement was skipped due to the max_mesh_elements constraint.
            If so, the refined mesh and new graph are the same as the input mesh and graph, respectively.

        """
        # 1. Predict values, clip to generate sizing field
        features = mesh_image.features.to(self.device)
        predictions = self._predict(features, train=False)
        # filter valid predictions
        predictions = predictions[0, mesh_image.is_mesh]
        sizing_field = self._clip_sizing_field(predictions)

        # 2. Estimate number of elements in the new mesh
        if self.max_mesh_elements is not None:
            from src.algorithms.amber.amber_util import edge_length_to_volume

            pixel_volume = mesh_image.pixel_volume
            new_element_volumes = edge_length_to_volume(sizing_field, mesh.dim())
            # allow for some extra elements in the estimate
            approx_new_num_elements = np.sum(pixel_volume / (2*new_element_volumes))
            if approx_new_num_elements > self.max_mesh_elements:
                # no refinement if the new mesh would be too large
                return predictions, mesh, False

        # 3. Construct new mesh from sizing field
        from src.environments.domains.gmsh_util import update_mesh
        new_mesh = update_mesh(old_mesh=mesh.mesh,
                               sizing_field=sizing_field,
                               gmsh_kwargs=self.gmsh_kwargs,
                               sizing_field_positions=mesh_image.mesh_positions.T)

        return predictions, new_mesh, True

    def _clip_sizing_field(self, predictions: torch.Tensor) -> np.ndarray:
        """
        Clips and detaches the predictions to a numpy array.
        Args:
            predictions:

        Returns:

        """
        sizing_field = np.clip(detach(predictions), self.gmsh_kwargs.get("min_sizing_field"), np.infty)
        return sizing_field

    def _get_initial_metrics(self) -> ValueDict:
        return {"min_sizing_field": self.gmsh_kwargs.get("min_sizing_field"),
                "num_batch_images": self.num_batch_images,
                "max_mesh_elements": self.max_mesh_elements,
                "network_parameters": count_parameters(self.model)
                }

    @property
    def running_normalizer(self) -> AbstractRunningNormalizer:
        """
        Wrapper for the environment normalizer.
        Returns:

        """
        return self._running_normalizer

    @property
    @lru_cache(maxsize=1)
    def num_batch_images(self):
        if self.batch_size == "auto":
            if self.use_gpu:
                # estimate the maximum batch size that fits on the GPU
                from src.hmpn.hmpn_util.calculate_max_batch_size import estimate_max_batch_size
                example_image = self.train_loader.first.observation.features
                example_image = example_image.to(self.device)
                max_batch_size = estimate_max_batch_size(
                    model=self.model,
                    input_sample=example_image,
                    device=self.device,
                    verbose=False,
                    rel_tolerance=0.16
                )
                return max_batch_size
            else:
                # return a default batch size of 8 on cpu
                return 8
        else:
            return self.batch_size

    #################
    # save and load #
    #################

    def _load_architecture_from_path(self, state_dict_path: Path) -> AbstractArchitecture:
        return AbstractArchitecture.load_from_path(
            state_dict_path=state_dict_path,
            in_features=self.in_features,
            out_features=1,  # predict a scalar per pixel
            device=self.device,
        )

    ####################
    # additional plots #
    ####################

    def additional_plots(self, iteration: int) -> Dict[Key, go.Figure]:
        """
        May provide arbitrary functions here that are used to draw additional plots.
        Args:
            iteration: Algorithm iteration that this function was called at
        Returns: A dictionary of {plot_name: plot}, where plot_function is any function that takes
          this algorithm at a current point as an argument, and returns a plotly figure.

        """
        from util.torch_util.torch_util import detach

        additional_plots = {}
        for evaluation_idx in self.evaluation_idxs:
            if len(self.validation_loader) <= evaluation_idx:
                continue  # loader does not contain this index

            data = self.validation_loader[evaluation_idx]
            data: MeshRefinementData
            mesh_image: MeshImage = copy.deepcopy(data.observation)
            mesh = data.mesh
            is_mesh = mesh_image.is_mesh

            fem_idx = data.fem_idx
            fem_problem = self.validation_loader.fem_problems[fem_idx]
            plot_boundary = fem_problem.plot_boundary

            inference_tuple = self._inference_step(mesh_image=mesh_image, mesh=mesh)
            predictions, refined_mesh, refinement_success = inference_tuple

            label_grid = detach(mesh_image.label_grid)

            prediction_grid = np.zeros(np.prod(label_grid.shape))
            prediction_grid[is_mesh] = detach(predictions)
            prediction_grid = prediction_grid.reshape(label_grid.shape)

            pixel_features = detach(mesh_image.features)

            solution = fem_problem.calculate_solution(refined_mesh)
            current_plots = get_prediction_image_plots(mesh_image=mesh_image,
                                                       prediction_grid=prediction_grid)
            current_plots |= get_learner_plots(learner_mesh=refined_mesh,
                                               plot_boundary=plot_boundary,
                                               solution=solution,
                                               refinement_success=refinement_success,
                                               )
            current_plots = prefix_keys(current_plots, prefix=f"idx{evaluation_idx}")
            additional_plots = additional_plots | current_plots

            if self._initial_plots:
                feature_plots = get_feature_image_plots(feature_names=self.in_feature_names,
                                                        mesh_image=mesh_image)
                feature_plots = prefix_keys(feature_plots, prefix=f"idx{evaluation_idx}_features")

                expert_mesh = self.validation_loader.expert_meshes[data.fem_idx]
                expert_plots = get_reference_plots(
                    reference_mesh=expert_mesh,
                    fem_problem=copy.deepcopy(fem_problem),
                    reference_name="Expert",
                )
                expert_plots = prefix_keys(expert_plots, prefix=f"idx{evaluation_idx}_expert")
                reconstructed_mesh = self.validation_loader.get_reconstructed_mesh(fem_idx)
                reconstructed_plots = get_reference_plots(
                    reference_mesh=reconstructed_mesh,
                    fem_problem=copy.deepcopy(fem_problem),
                    reference_name="Reconstructed",
                )
                reconstructed_plots = prefix_keys(reconstructed_plots, prefix=f"idx{evaluation_idx}_reconstructed")

                additional_plots = additional_plots | feature_plots | expert_plots | reconstructed_plots

        if self._initial_plots:
            self._initial_plots = False
        return additional_plots

    def get_final_values(self) -> ValueDict:
        """
        Returns a dictionary of values that are logged at the end of the training.
        Here, this includes evaluations on the test buffer.
        Returns:

        """
        final_values = self._evaluate_dataloader(dataloader=self.test_loader, description="test")
        final_values = filter_scalars(final_values)
        return final_values
