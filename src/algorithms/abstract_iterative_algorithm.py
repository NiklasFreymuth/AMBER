from abc import ABC, abstractmethod
from pathlib import Path

import plotly.graph_objects as go

from src.algorithms.util.normalizers.abstract_running_normalizer import (
    AbstractRunningNormalizer,
)
from src.modules.abstract_architecture import AbstractArchitecture
from util import keys as Keys
from util.save_and_load.checkpoint import Checkpoint
from util.types import *


class AbstractIterativeAlgorithm(ABC):
    def __init__(self, config: ConfigDict) -> None:
        """
        Initializes the iterative algorithm using the full config used for the experiment.
        Args:
            config: A (potentially nested) dictionary containing the "params" section of the section in the .yaml file
                used by cw2 for the current run.
        Returns:

        """
        self._config = config
        self._algorithm_config = config.get("algorithm")

        from torch import device
        from torch.cuda import is_available as cuda_is_available

        self.use_gpu = self.algorithm_config.get("use_gpu") and cuda_is_available()
        if self.use_gpu:
            self._device: device = device("cuda" if cuda_is_available() else "cpu")

            # try to reduce the max split size to decrease fragmentation and avoid OOM errors
            max_split_size_mb = self.algorithm_config.get("max_gpu_split_size_mb")
            if max_split_size_mb is not None:
                import os
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = f'max_split_size_mb:{max_split_size_mb}'
        else:
            self._device = "cpu"

    ##############
    # properties #
    ##############
    @property
    def config(self) -> ConfigDict:
        return self._config

    @property
    def algorithm_config(self) -> ConfigDict:
        return self._algorithm_config

    @property
    def network_config(self) -> ConfigDict:
        return self._algorithm_config.get("network")

    @property
    def verbose(self) -> bool:
        return self.algorithm_config.get("verbose", False)

    @property
    def device(self) -> str:
        return self._device

    @property
    def batch_size(self) -> int:
        return self.algorithm_config.get("batch_size")

    ###########
    # methods #
    ###########

    def fit_and_evaluate(self) -> ValueDict:
        """
        Trains the algorithm for a single iteration, evaluates its performance and subsequently organizes and provides
        metrics, losses, plots etc. of the fit and evaluation.
        This is the main method that should be called to train and evaluate the algorithm.
        Returns:

        """
        if self.use_gpu:
            from torch import cuda
            cuda.empty_cache()
        train_values = self.fit_iteration()
        evaluation_values = self.evaluate()

        # collect and organize values
        full_values = train_values | evaluation_values

        value_dict = filter_scalars(full_values)
        return value_dict

    @abstractmethod
    def fit_iteration(self) -> ValueDict:
        """
        Train your algorithm for a single iteration. This can e.g., be a single epoch of neural network training,
        a policy update step, or something more complex. Just see this as the outermost for-loop of your algorithm.

        Returns: May return an optional dictionary of values produced during the fit. These may e.g., be statistics
        of the fit such as a training loss.

        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self) -> ValueDict:
        """
        Evaluate given input data and potentially auxiliary information to create a dictionary of resulting values.
        What kind of things are scored/evaluated depends on the concrete algorithm.
        Args:

        Returns: A dictionary with different values that are evaluated from the given input data. May e.g., the
        accuracy of the model.

        """
        raise NotImplementedError

    @property
    def architecture(self) -> AbstractArchitecture:
        raise NotImplementedError("AbstractIterativeAlgorithm does not implement architecture")

    @property
    def learning_rate_scheduler(self):
        return self.architecture.learning_rate_scheduler

    @property
    def running_normalizer(self) -> Optional[AbstractRunningNormalizer]:
        """
        Wrapper for the environment normalizer. May be None if no normalization is used.
        Returns:

        """
        return None

    #################
    # save and load #
    #################
    def save_checkpoint(
            self,
            directory: str,
            iteration: Optional[int] = None,
            is_final_save: bool = False,
            is_initial_save: bool = False,
    ) -> None:
        """
        Save the current state of the algorithm to the given directory. This includes the policy, the optimizer, and
        the environment normalizer (if applicable).
        Args:
            directory:
            iteration:
            is_final_save:
            is_initial_save:

        Returns:

        """
        checkpoint_path = Path(directory)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        policy_save_path = self.architecture.save(
            destination_folder=checkpoint_path,
            file_index=Keys.FINAL if is_final_save else iteration,
            save_kwargs=is_initial_save,
        )
        if self.running_normalizer is not None:
            import util.save_and_load.save_and_load_keys as K

            # load environment normalizer if available. Load from same iteration as network
            normalizer_save_path = policy_save_path.with_name(
                policy_save_path.stem + f"_{K.NORMALIZATION_PARAMETER_SUFFIX}.pkl"
            )
            self.running_normalizer.save(destination_path=normalizer_save_path)

    def load_from_checkpoint(self, checkpoint_config: ConfigDict) -> Checkpoint:
        """
        Loads the algorithm state from the given checkpoint path/experiment configuration name.
        May be used at the start of the algorithm to resume training.
        Args:
            checkpoint_config: Dictionary containing the configuration of the checkpoint to load. Includes
                checkpoint_path: Path to a checkpoint folder of a previous execution of the same algorithm
                iteration: (iOptional[int]) The iteration to load. If not provided, will load the last available iter
                repetition: (int) The algorithm repetition/seed to load. If not provided, will load the first repetition

        Returns:

        """
        import os

        import util.save_and_load.save_and_load_keys as K

        # get checkpoint path and iteration
        experiment_name = checkpoint_config.get("experiment_name")
        iteration = checkpoint_config.get("iteration")
        repetition = checkpoint_config.get("repetition")
        if repetition is None:
            repetition = 0  # default to first repetition

        # format checkpoint path
        if "__" in experiment_name:
            # grid experiments, add the main experiment as the first part of the path
            experiment_name = os.path.join(experiment_name[0: experiment_name.find("__")], experiment_name)

        if checkpoint_config.get("load_root_dir") is None:
            load_root_dir = K.REPORT_FOLDER
        else:
            load_root_dir = checkpoint_config["load_root_dir"]

        checkpoint_path = os.path.join(
            load_root_dir,
            experiment_name,
            "log",
            f"rep_{repetition:02d}",
            K.SAVE_DIRECTORY,
        )
        checkpoint_path = Path(checkpoint_path)
        assert checkpoint_path.exists(), f"Checkpoint path {checkpoint_path} does not exist"

        # load state dict for network
        if iteration is not None:  # if iteration is given, load the corresponding file
            file_name = f"{K.TORCH_SAVE_FILE}{iteration:04d}.pt"
        else:
            file_name = f"{K.TORCH_SAVE_FILE}_{Keys.FINAL}.pt"
            if not (checkpoint_path / file_name).exists():
                # if final.pkl does not exist, load the last iteration instead
                file_name = sorted(list(checkpoint_path.glob("*.pt")))[-1].name
        state_dict_path = checkpoint_path / file_name
        architecture = self._load_architecture_from_path(state_dict_path=state_dict_path)
        architecture.to(self.device)
        # load environment normalizer if available. Load from same iteration as network
        normalizer_path = state_dict_path.with_name(state_dict_path.stem + f"_{K.NORMALIZATION_PARAMETER_SUFFIX}.pkl")
        if normalizer_path.exists():
            normalizer = AbstractRunningNormalizer.load(checkpoint_path=normalizer_path)
            normalizer.to(self.device)
        else:
            normalizer = None

        return Checkpoint(architecture=architecture, normalizer=normalizer)

    def _load_architecture_from_path(self, state_dict_path: Path) -> AbstractArchitecture:
        raise NotImplementedError("AbstractIterativeAlgorithm does not implement _load_architecture_from_path()")

    ####################################
    # additional plots and evaluations #
    ####################################

    def additional_plots(self, iteration: int) -> Dict[Key, go.Figure]:
        """
        May provide arbitrary functions here that are used to draw additional plots.
        Args:
            iteration: The algorithm iteration this function was called at
        Returns: A dictionary of {plot_name: plot}, where plot_function is any function that takes
          this algorithm at a current point as an argument, and returns a plotly figure.

        """
        raise NotImplementedError("AbstractIterativeAlgorithm does not implement additional_plots()")

    def get_final_values(self) -> ValueDict:
        """
        Returns a dictionary of values that are to be stored as final values of the algorithm.
        Returns:

        """
        raise NotImplementedError("AbstractIterativeAlgorithm does not implement get_final_values()")


def filter_scalars(full_values) -> dict:
    """
    Filters the scalar values from the full values dict. Save them in a separate dict and return it.
    Args:
        full_values:

    Returns:

    """
    import util.keys as Keys
    scalars = {}
    value_dict = {}
    for key, value in full_values.items():
        if key in [
            Keys.FIGURES,
            Keys.VIDEO_ARRAYS,
        ]:  # may either be a single figure, or a list of figures
            value_dict[key] = value
        elif isinstance(value, list):
            if len(value) > 0:  # list is not empty
                scalars[key] = value[-1]
        else:
            scalars[key] = value
    value_dict[Keys.SCALARS] = scalars
    return value_dict
