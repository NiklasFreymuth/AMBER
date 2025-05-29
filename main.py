import logging
import os
import sys
import traceback
import warnings
from typing import List

import hydra
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from omegaconf import DictConfig, OmegaConf

from src.initialization import initialize
from src.initialization.init_config import load_omega_conf_resolvers
from src.logger.progress_bar import CustomProgressBar

# full stack trace
os.environ["HYDRA_FULL_ERROR"] = "1"

# register OmegaConf resolver for hydra
load_omega_conf_resolvers()

warnings.filterwarnings("ignore", category=UserWarning)

logging.getLogger("skfem").setLevel(logging.WARNING)  # Only show warnings and errors


@hydra.main(version_base=None, config_path="config", config_name="training_config")
def train(config: DictConfig) -> None:
    try:
        logging.getLogger("skfem").setLevel(logging.WARNING)  # Only show warnings and errors

        print(OmegaConf.to_yaml(config, resolve=True))
        exp_root = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        if config.trainer.get("matmul_precision", None) is not None:
            torch.set_float32_matmul_precision(config.trainer.matmul_precision)

        initialization_return = initialize(config=config)

        logger = initialization_return.wandb_logger
        callbacks = get_callbacks(config, exp_root, logger, checkpoint_frequency=config.trainer.checkpoint_frequency)
        trainer_config = config.trainer
        trainer = Trainer(
            logger=logger,  # Use the wandb logger
            callbacks=callbacks,  # Checkpointing callback
            default_root_dir=exp_root,  # Where to save logs and checkpoints
            max_epochs=trainer_config.max_epochs,
            accelerator=trainer_config.accelerator,
            devices=trainer_config.devices,
            precision=trainer_config.precision,
            accumulate_grad_batches=trainer_config.accumulate_grad_batches,
            check_val_every_n_epoch=trainer_config.check_val_every_n_epoch,
            enable_checkpointing=trainer_config.enable_checkpointing,
            enable_progress_bar=True,
            enable_model_summary=False,
        )

        # Start the training
        dataloaders = initialization_return.dataloaders
        algorithm: LightningModule = initialization_return.algorithm

        if config.trainer.get("torch_compile", False):
            # Compile the algorithm for performance optimization
            # Note: This is a placeholder. The actual compilation method may vary.
            algorithm = torch.compile(algorithm)  # Todo: Test!
        trainer.fit(algorithm, train_dataloaders=dataloaders.get("train"), val_dataloaders=dataloaders.get("val"))

        trainer.test(algorithm, dataloaders=dataloaders.get("test"))
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


def get_callbacks(config: DictConfig, exp_root: str, wandb_logger, checkpoint_frequency: int) -> List["Callback"]:
    print(exp_root)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(exp_root, "checkpoints"),  # Directory to save checkpoints
        filename="checkpoint-{epoch:02d}",  # Checkpoint filename format
        every_n_epochs=checkpoint_frequency,  # Save checkpoint every K epochs
        save_top_k=-1,  # Save all checkpoints
        save_last=True,  # Optionally save the most recent model
    )
    callbacks = [CustomProgressBar()]
    if config.trainer.enable_checkpointing:
        callbacks.append(checkpoint_callback)
    if wandb_logger:
        learning_rate_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(learning_rate_monitor)
    return callbacks


if __name__ == "__main__":
    train()
