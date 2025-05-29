import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from src.logger.custom_wandb_logger import CustomWandBLogger, reset_wandb_env


def get_wandb_logger(config: DictConfig) -> CustomWandBLogger:
    """
    Create a wandb logger with the given config and algorithm.
    Args:
        logger_config:

    Returns: A wandb logger to use.
    """
    reset_wandb_env()

    logger_config = config.logger

    wandb_config = logger_config.wandb
    project_name = wandb_config.get("project_name")
    environment_name = wandb_config.task_name

    if wandb_config.get("task_name") is not None:
        project_name = project_name + "_" + wandb_config.get("task_name")
    elif environment_name is not None:
        project_name = project_name + "_" + environment_name
    else:
        # no further specification of the project, just use the initial project_name
        project_name = project_name

    groupname = wandb_config.get("group_name")[-127:]
    job_type = wandb_config.get("job_type")[-63:]
    runname = job_type[-63:] + "_" + wandb_config.get("run_name")[-63:]

    tags = wandb_config.get("tags", [])
    if tags is None:
        tags = []
    if "idx" in config:
        runname = (runname + "_i" + str(config.idx))[-127:]
        job_type = (job_type + "_i" + str(config.idx))[-63:]
        tags.append("i" + str(config.idx))
    if "_version" in config:
        runname = (runname + "_v" + str(config._version))[-127:]
        job_type = (job_type + "_v" + str(config._version))[-63:]
        tags.append("v" + str(config._version))

    entity = wandb_config.get("entity")

    start_method = wandb_config.get("start_method")
    settings = wandb.Settings(start_method=start_method) if start_method is not None else None

    # Initialize WandB logger
    wandb_logger = CustomWandBLogger(
        config=OmegaConf.to_container(config, resolve=True),
        project=project_name,  # Name of your WandB project
        name=runname,  # Name of the current run
        group=groupname,  # Group name for the run
        tags=tags,  # List of tags for your run
        entity=entity,  # WandB username or team name
        settings=settings,  # Optional WandB settings
        job_type=job_type,  # Name of your experiment
        log_model=False,
        save_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
    )
    return wandb_logger
