from functools import lru_cache
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from tqdm import tqdm

from src.algorithms.abstract_iterative_algorithm import AbstractIterativeAlgorithm, filter_scalars
from src.algorithms.amber.amber_data_generation import get_train_val_loaders
from src.algorithms.amber.amber_metrics import get_similarity_metrics, get_mesh_similarity_metrics
from src.algorithms.amber.amber_util import calculate_loss
from src.algorithms.amber.amber_visualization import get_learner_plots, get_reference_plots
from src.algorithms.amber.architectures.supervised_hmpn import SupervisedHMPN
from src.algorithms.amber.dataloader.amber_data import AMBERData
from src.algorithms.amber.dataloader.amber_dataloader import AMBERDataLoader
from src.algorithms.amber.dataloader.amber_online_dataloader import AMBEROnlineDataLoader
from src.algorithms.amber.mesh_wrapper import MeshWrapper
from src.algorithms.util.normalizers.abstract_running_normalizer import AbstractRunningNormalizer
from src.algorithms.util.normalizers.dummy_running_normalizer import DummyRunningNormalizer
from src.algorithms.util.normalizers.graph_running_normalizer import GraphRunningNormalizer
from src.modules.abstract_architecture import AbstractArchitecture
from util.function import add_to_dictionary, prefix_keys, safe_mean
from util.torch_util.torch_util import detach, count_parameters
from util.types import *


class AMBER(AbstractIterativeAlgorithm):
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
        self.buffer_add_frequency = supervised_config.get("buffer_add_frequency")
        self.buffer_add_strategy = supervised_config.get("buffer_add_strategy")
        self.inference_steps = supervised_config.get("inference_steps")
        self.max_buffer_mesh_depth = supervised_config.get("max_buffer_mesh_depth")
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
                checkpoint.architecture, SupervisedHMPN
            ), f"checkpoint must contain a GraphDQNPolicy, given type: '{type(checkpoint.architecture)}'"
            self._model = checkpoint.architecture
            self._running_normalizer = checkpoint.normalizer
        else:
            self._model = self._build_model()
            self._running_normalizer = self._build_normalizer(normalizer_config=supervised_config.get("normalizer"))
            [self._running_normalizer.update_observation_normalizers(graph) for graph in self.train_loader.observations]

    def _build_model(self) -> SupervisedHMPN:
        return SupervisedHMPN(
            example_graph=self.train_loader.first.graph,
            network_config=self.network_config,
            use_gpu=self.use_gpu,
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
            first_graph = self.train_loader.first.graph
            num_node_features = first_graph.x.shape[1]
            num_edge_features = first_graph.edge_attr.shape[1]
            if hasattr(first_graph, "u") and first_graph.u is not None:
                num_global_features = first_graph.u.shape[1]
                normalize_globals = True
            else:
                num_global_features = None
                normalize_globals = False
            running_normalizer = GraphRunningNormalizer(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                num_global_features=num_global_features,
                normalize_nodes=normalize_observations,
                normalize_edges=normalize_observations,
                normalize_globals=normalize_observations and normalize_globals,
                observation_clip=observation_clip,
                device=self.device,
            )
        else:
            running_normalizer = DummyRunningNormalizer()
        return running_normalizer

    @property
    def train_loader(self) -> AMBEROnlineDataLoader:
        return self._train_loader

    @property
    def validation_loader(self) -> AMBERDataLoader:
        return self._validation_loader

    @property
    def test_loader(self) -> AMBERDataLoader:
        return self._test_loader

    @property
    def model(self) -> AbstractArchitecture:
        return self._model

    @property
    def architecture(self) -> AbstractArchitecture:
        return self.model

    def fit_iteration(self) -> ValueDict:
        metric_dict_list = {}
        for batch_idx in tqdm(range(self.batches_per_iteration)):
            train_metrics = self._train_batch()
            train_metrics = prefix_keys(train_metrics, prefix="train")
            metric_dict_list = add_to_dictionary(metric_dict_list, new_scalars=train_metrics)

            if self.buffer_add_frequency > 0 and batch_idx % self.buffer_add_frequency == 0:
                online_metrics = self._add_online_data()
                online_metrics = prefix_keys(online_metrics, prefix="online")
                metric_dict_list = add_to_dictionary(metric_dict_list, new_scalars=online_metrics)

        mean_metrics = {key + "_mean": safe_mean(value) for key, value in metric_dict_list.items()}
        metrics = mean_metrics
        return metrics

    def _train_batch(self) -> ValueDict:
        self.model.train()
        graph_batch = self.train_loader.get_batch(num_nodes=self.num_batch_nodes)

        graph_batch = graph_batch.to(self.device)
        predictions = self._predict(graph_batch, train=True)

        loss = calculate_loss(
            predictions=predictions,
            labels=graph_batch.y,
            loss_type=self.loss_type,
        )

        self.model.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._max_grad_norm)  # Clip grad norm
        self.model.optimizer.step()
        # log metrics and training scalars

        with torch.no_grad():
            nodes_per_batch = graph_batch.ptr[1:] - graph_batch.ptr[:-1]
            nodes_per_batch = detach(nodes_per_batch)
            if self.loss_type == "log_mse":
                predictions = torch.exp(predictions)
                if torch.any(torch.isnan(predictions)):
                    import warnings
                    warnings.warn("Predictions contain NaN values. This may be due to a too large learning rate.")
                    predictions[torch.isnan(predictions)] = 100
            differences = torch.abs(predictions - graph_batch.y)
            train_scalars = {
                "loss": loss.item(),
                "min_dif": torch.min(differences).item(),
                "max_dif": torch.max(differences).item(),
                "mean_dif": torch.mean(differences).item(),
                "mean_nodes": np.mean(nodes_per_batch),
                "min_nodes": np.min(nodes_per_batch),
                "max_nodes": np.max(nodes_per_batch),
                "total_edges": graph_batch.num_edges,
                "total_nodes": graph_batch.num_nodes,
                "total_graphs": graph_batch.ptr.shape[0] - 1,
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
        predictions = predictions.flatten()

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

    def _add_online_data(self) -> ValueDict:
        """
        Predict a sizing field for an element of the buffer, use it to create a new mesh,
        and add this mesh with interpolated sizing field to the buffer.
        Returns:

        """
        self.model.train(False)

        # select point from the dataloader
        data: AMBERData = self.train_loader.get_data_point(max_depth=self.max_buffer_mesh_depth - 1,
                                                           strategy=self.buffer_add_strategy)
        fem_idx = data.fem_idx
        mesh_to_graph_fn = partial(self.train_loader.get_observation, fem_idx=fem_idx)
        graph: Union[Data, HeteroData] = copy.deepcopy(data.graph)
        mesh = data.mesh

        # perform a single inference step
        _, refined_mesh, new_graph, success = self._inference_step(
            graph=graph,
            mesh=mesh,
            mesh_to_graph_fn=mesh_to_graph_fn,
        )

        if success:
            self.running_normalizer.update_observation_normalizers(new_graph)

            new_depth = data.refinement_depth + 1
            _ = self.train_loader.add_data(
                fem_idx=fem_idx,
                mesh=refined_mesh,
                refinement_depth=new_depth,
                graph=new_graph,
            )

            metrics = {
                "new_depth": new_depth,
                "new_num_elements": refined_mesh.nelements,
            }
        else:
            metrics = {
                "new_depth": data.refinement_depth,
                "new_num_elements": mesh.nelements,
            }

        metrics["refinement_success"] = int(success)
        metrics["buffer_size"] = len(self.train_loader)
        # can be different from max_buffer_size due to protected data

        return metrics

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

    def _evaluate_dataloader(self, dataloader: AMBERDataLoader, description: str) -> ValueDict:
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
            data: AMBERData
            fem_idx: int = data.fem_idx
            mesh_to_graph_fn: callable = partial(dataloader.get_observation, fem_idx=fem_idx)
            expert_mesh = dataloader.expert_meshes[fem_idx]
            reconstructed_mesh = dataloader.get_reconstructed_mesh(fem_idx)
            graph: Union[Data, HeteroData] = copy.deepcopy(data.graph)
            mesh = data.mesh  # initial mesh

            # get comparison between expert and initial mesh. This is constant for the same dataset
            initial2expert_similarity_metrics = get_mesh_similarity_metrics(mesh1=mesh,
                                                                            mesh2=expert_mesh)
            initial2expert_similarity_metrics = prefix_keys(initial2expert_similarity_metrics, prefix="exp_init")
            add_to_dictionary(metric_dict_list, new_scalars=initial2expert_similarity_metrics)

            # perform self.inference_steps iterative refinement steps.
            cumulative_elements = mesh.nelements
            for step in range(self.inference_steps):
                (
                    predictions,
                    refined_mesh,
                    new_graph,
                    refinement_success,
                ) = self._inference_step(
                    graph=graph,
                    mesh=mesh,
                    mesh_to_graph_fn=mesh_to_graph_fn,  # mesh_to_graph_fn contains labelling from expert mesh
                )

                is_last_step = step == self.inference_steps - 1
                step_name = "" if is_last_step else f"step{step}"

                # online metrics
                cumulative_elements += refined_mesh.nelements
                differences = detach(torch.abs(predictions - graph.y))
                online_metrics = {
                    "sf_MAE": np.mean(differences),
                    "sf_MSE": np.mean(differences ** 2),
                    "refinement_success": int(refinement_success),
                    "cum_elements": cumulative_elements,
                    "cur_elements": refined_mesh.nelements,
                    "min_sf": np.min(self._clip_sizing_field(predictions)),
                }
                if not is_last_step:
                    online_metrics = prefix_keys(online_metrics, prefix=step_name)
                metric_dict_list = add_to_dictionary(metric_dict_list, new_scalars=online_metrics)

                if refinement_success:
                    # calculate metrics for the refined mesh
                    mesh_similarity_metrics = get_similarity_metrics(refined_mesh=refined_mesh,
                                                                     expert_mesh=expert_mesh,
                                                                     reconstructed_mesh=reconstructed_mesh if is_last_step else None)
                    if not is_last_step:
                        mesh_similarity_metrics = prefix_keys(mesh_similarity_metrics, prefix=step_name)
                    metric_dict_list = add_to_dictionary(metric_dict_list, new_scalars=mesh_similarity_metrics)

                    # prepare for next step
                    graph = new_graph
                    mesh = refined_mesh
                else:
                    # mesh can not currently be refined by our method. Break the eval for this mesh
                    for remaining_step in range(step + 1, self.inference_steps):
                        remaining_dict = {"refinement_success": 0}
                        if not remaining_step == self.inference_steps - 1:
                            remaining_dict = prefix_keys(remaining_dict, prefix=f"step{remaining_step}")
                        metric_dict_list = add_to_dictionary(metric_dict_list, new_scalars=remaining_dict)
                    break
        evaluation_dict = {key: safe_mean(value) for key, value in metric_dict_list.items()}
        evaluation_dict = prefix_keys(evaluation_dict, description)

        return evaluation_dict

    def _inference_step(
            self,
            graph: Union[Data, HeteroData],
            mesh: MeshWrapper,
            mesh_to_graph_fn: callable,
    ) -> Tuple[torch.Tensor, MeshWrapper, Union[Data, HeteroData], bool]:
        """
        Perform a single step of iterative inference, i.e. predict a sizing field, refine the mesh, and return the new
        graph and mesh.
        Args:
            graph: The graph to predict the sizing field for
            mesh: The scikit FEM mesh to refine
            mesh_to_graph_fn: A function to convert a mesh to a graph. Has to be a function of only the mesh, and must
                return a torch geometric graph. Internally, this function should use the FEM problem to calculate the
                solution on the mesh and convert it to a graph.

        Returns: A 4-tuple of
        * the predictions on the old graph,
        * the refined mesh, wrapped in a CachedMeshWrapper,
        * the new graph on this mesh, and
        * refinement success, i.e., whether the refinement was skipped due to the max_mesh_elements constraint.
            If so, the refined mesh and new graph are the same as the input mesh and graph, respectively.


        """
        # 1. Predict values, clip to generate sizing field
        graph = graph.to(self.device)
        predictions = self._predict(graph, train=False)
        sizing_field = self._clip_sizing_field(predictions)

        # 2. Estimate number of elements in the new mesh
        if self.max_mesh_elements is not None:
            from src.algorithms.amber.amber_util import edge_length_to_volume

            simplex_volumes = mesh.get_simplex_volumes()
            new_element_volumes = edge_length_to_volume(sizing_field, mesh.dim())
            approx_new_num_elements = np.sum(simplex_volumes / new_element_volumes)
            if approx_new_num_elements > self.max_mesh_elements:
                # no refinement if the new mesh would be too large
                return predictions, mesh, graph, False

        # 3. Construct new mesh from sizing field
        from src.environments.domains.gmsh_util import update_mesh
        new_mesh = update_mesh(old_mesh=mesh.mesh, sizing_field=sizing_field, gmsh_kwargs=self.gmsh_kwargs)

        # 4. Get torch graph for the refined mesh and its solution
        graph = mesh_to_graph_fn(mesh=new_mesh)
        return predictions, new_mesh, graph, True

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
                "num_batch_nodes": self.num_batch_nodes,
                "max_mesh_elements": self.max_mesh_elements,
                "avg_expert_elements": np.mean([mesh.nelements for mesh in self.train_loader.expert_meshes]),
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
    def num_batch_nodes(self):
        if self.batch_size == "auto":
            if self.use_gpu:
                # estimate the maximum batch size that fits on the GPU
                from src.hmpn.hmpn_util.calculate_max_batch_size import estimate_max_batch_size
                example_graph = copy.deepcopy(self.train_loader.first.graph)
                max_batch_size = estimate_max_batch_size(
                    model=self.model,
                    input_sample=example_graph,
                    device=self.device,
                    verbose=False,
                    rel_tolerance=0.16,
                )
                num_nodes = example_graph.num_nodes
                return max_batch_size * num_nodes
            else:
                # return a default of 100000 nodes/elements for now
                return 100000
        else:
            return self.max_mesh_elements * self.batch_size

    #################
    # save and load #
    #################

    def _load_architecture_from_path(self, state_dict_path: Path) -> AbstractArchitecture:
        return AbstractArchitecture.load_from_path(
            state_dict_path=state_dict_path, example_graph=self.train_loader.first.graph, device=self.device
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
                continue
            data = self.validation_loader[evaluation_idx]
            fem_idx = data.fem_idx
            mesh_to_graph_fn = partial(self.validation_loader.get_observation, fem_idx=fem_idx)
            fem_problem = self.validation_loader.fem_problems[fem_idx]
            graph: Union[Data, HeteroData] = copy.deepcopy(data.graph)
            mesh = data.mesh
            plot_boundary = fem_problem.plot_boundary

            final_suffix = ""
            refinement_success = True
            for step in range(self.inference_steps):
                inference_tuple = self._inference_step(
                    graph=graph,
                    mesh=mesh,
                    mesh_to_graph_fn=mesh_to_graph_fn,
                )
                predictions, refined_mesh, new_graph, refinement_success = inference_tuple

                step_name = "" if step == self.inference_steps - 1 else f"_step{step}"

                if not refinement_success:
                    # mesh can not currently be refined by our method
                    final_suffix = step_name
                    break

                current_plots = get_learner_plots(learner_mesh=mesh,
                                                  plot_boundary=plot_boundary,
                                                  labels=detach(graph.y),
                                                  predicted_sizing_field=self._clip_sizing_field(predictions),
                                                  )
                current_plots = prefix_keys(current_plots,
                                            prefix=f"idx{evaluation_idx}{step_name}")
                additional_plots = additional_plots | current_plots

                # prepare for next step
                graph = new_graph
                mesh = refined_mesh

            solution = fem_problem.calculate_solution(mesh)
            current_plots = get_learner_plots(learner_mesh=mesh,
                                              plot_boundary=plot_boundary,
                                              labels=detach(graph.y),
                                              solution=solution,
                                              refinement_success=refinement_success,
                                              )
            current_plots = prefix_keys(current_plots, prefix=f"idx{evaluation_idx}{final_suffix}")
            additional_plots = additional_plots | current_plots

            if self._initial_plots:
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

                additional_plots = additional_plots | expert_plots
                additional_plots = additional_plots | reconstructed_plots

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
