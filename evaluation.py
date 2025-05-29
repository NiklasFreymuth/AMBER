import logging
import os
import sys
import traceback
import warnings
from typing import Dict, List

# deterministic cublas implementation (for reproducibility)
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import hydra
import numpy as np
import torch
from lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.algorithm import create_algorithm
from src.algorithm.dataloader import get_dataloader, get_datasets
from src.initialization import initialize_seed
from src.initialization.init_config import initialize_config, load_omega_conf_resolvers
from src.logger.evaluation_logger import EvaluationLogger

# full stack trace
os.environ["HYDRA_FULL_ERROR"] = "1"

# register OmegaConf resolver for hydra
load_omega_conf_resolvers()

warnings.filterwarnings("ignore", category=UserWarning)

logging.getLogger("skfem").setLevel(logging.WARNING)  # Only show warnings and errors


@hydra.main(version_base=None, config_path="config", config_name="test_config")
def evaluation(config: DictConfig) -> None:
    try:
        exp_root = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        initialize_config(config)  # Apply SLURM-related settings to the configuration
        initialize_seed(config.seed)  # Set random seed for reproducibility

        datasets = get_datasets(algorithm_config=config.algorithm, task_config=config.task)

        # Create dataloaders for each dataset mode (e.g., "train", "val", "test)
        dataloaders = {
            dataset_mode: get_dataloader(algorithm_config=config.algorithm, dataset=dataset, is_train=dataset_mode == "train")
            for dataset_mode, dataset in datasets.items()
        }

        if config.trainer.get("matmul_precision", None) is not None:
            torch.set_float32_matmul_precision(config.trainer.matmul_precision)

        loading_path = config.loading.root_path
        if loading_path.endswith("/"):
            loading_path = loading_path[:-1]
        output_path = config.loading.output_path
        os.makedirs(output_path, exist_ok=True)
        # last folder is exp_name
        exp_name = loading_path.split("/")[-1]

        ckpt_queue = get_ckpts(loading_path, checkpoint_loading=config.loading.checkpoint)

        for ckpt_dict in tqdm(ckpt_queue, desc="Evaluating checkpoints", unit="checkpoint"):
            checkpoint_path = ckpt_dict["checkpoint_path"]
            job_type = ckpt_dict["job_type"]
            seed = ckpt_dict["seed"]
            algorithm = create_algorithm(
                algorithm_config=config.algorithm,
                train_dataset=datasets.get("train"),
                loading=True,
                checkpoint_path=checkpoint_path,
            )

            # this is the correct parent directory to load the algorithm and the checkpoints
            test_logger = EvaluationLogger(
                output_path=config.loading.output_path, exp_name=exp_name, save_figures=config.loading.save_figures, job_type=job_type, seed=seed
            )
            trainer = Trainer(
                logger=test_logger,  # results will be written to disk
                max_epochs=config.trainer.max_epochs,  # Max number of epochs for training
                accelerator=config.trainer.accelerator,  # what type of accelerator to use
                devices=config.trainer.devices,  # how many devices to use (if accelerator is not None)
                precision=config.trainer.precision,  # Precision setting (e.g., 16-bit)
                callbacks=None,
                default_root_dir=exp_root,  # dummy path, will be used multiple times presumably
                enable_checkpointing=False,  # no checkpointing
            )

            OmegaConf.set_struct(algorithm.sizing_field_damping, False)

            if config.get("last_step_damping").get("do_last_step_damping"):
                last_step_damping_config = config.last_step_damping
                min_sizing_field = algorithm.gmsh_kwargs.get("min_sizing_field")
                algorithm.max_mesh_elements = 1e6  # allow for a bunch more elements
                for last_step_damping in np.geomspace(
                    start=last_step_damping_config.start, stop=last_step_damping_config.stop, num=last_step_damping_config.num_steps
                ):
                    algorithm.sizing_field_damping.last_step_damping = float(last_step_damping)
                    algorithm.gmsh_kwargs["min_sizing_field"] = min_sizing_field * last_step_damping
                    test_logger.experiment.suffix = f"{last_step_damping:.2f}"

                    # Now, start the test
                    trainer.test(algorithm, dataloaders=dataloaders.get("test"))
            else:
                # No suffix, no adaptation to the algorithm needed
                trainer.test(algorithm, dataloaders=dataloaders.get("test"))
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


def get_ckpts(loading_path, checkpoint_loading: str) -> List[Dict[str, str]]:
    """
    Get the checkpoints to load from the loading path. The loading path should contain a folder for each job type, and
    each job type should contain a folder for each seed. Each seed folder should contain a checkpoints folder with the
    checkpoints to load. The checkpoints should be named according to the epoch they were saved at, or "last.ckpt".
    Args:
        loading_path:
        checkpoint_loading:

    Returns: A list of dictionaries, each containing the job type, seed, and checkpoint path of a given run.

    """
    ckpt_queue: List[Dict[str, str]] = []  # list of runs to execute one after the other.
    for job_type in os.listdir(loading_path):
        job_type_path = os.path.join(loading_path, job_type)
        for seed in os.listdir(job_type_path):
            seed_path = os.path.join(job_type_path, seed)
            checkpoint_path = os.path.join(seed_path, "checkpoints")
            if checkpoint_loading == "last":
                checkpoint_path = os.path.join(checkpoint_path, "last.ckpt")
            else:
                try:
                    checkpoint_epoch = int(checkpoint_loading)
                    checkpoint_path = os.path.join(checkpoint_path, f"checkpoint-epoch={checkpoint_epoch:02d}.ckpt")
                except:
                    raise ValueError(f"Invalid checkpoint: {checkpoint_loading}")

            ckpt_queue.append(
                {
                    "job_type": job_type,
                    "seed": seed,
                    "checkpoint_path": checkpoint_path,
                }
            )
    return ckpt_queue


if __name__ == "__main__":
    evaluation()
