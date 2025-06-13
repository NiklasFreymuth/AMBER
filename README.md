# AMBER

Code for [AMBER: Adaptive Mesh Generation by Iterative Mesh Resolution Prediction](http://arxiv.org/abs/2505.23663).

For the earlier [workshop version of AMBER](https://arxiv.org/abs/2406.14161), see the ```workshop``` branch.
# Getting Started

## Setting up the environment

### Mamba
This project uses [mamba](https://github.com/conda-forge/miniforge) / [conda](https://docs.conda.io/en/latest/) and pip for handling packages and dependencies.
To install mamba on Linux-like OSes use one of the commands below.

```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

For Windows please see the documentation in the link above or use (not recommended).
```
conda install -c conda-forge mamba
```

Afterward, you should be able to install all requirements using the commands below:

```
# for cpu use
mamba env create -f ./env/environment-cpu.yaml

# for gpu use
mamba env create -f ./env/environment-cuda.yaml

# Activate environment
mamba activate AMBER_neurips

wandb login  # login into wandb
pre-commit install  # install pre-commit for uniform formatting
```

### Test the environment
Test if everything works by running an experiment:

```bash
python main.py +_runs=debug
```


## Data
As part of AMBER, we propose six datasets, namely
* Poisson
* Laplace
* Airfoil
* Beam
* Console
* Mold

Poisson and Laplace are dynamically created during runtime.
The other datasets are provided under ./data/.


## Creating an experiment

Experiments are configured and distributed via hydra. For this, the folder `config` contains
a number of `.yaml` files that describe the configuration of the task to run.
The folder `_runs` contains individual *experiments*, each of which is a separate `.yaml`.

You can start an experiment, such as `debug` (in a corresponding `_runs/test/debug.yaml`), locally with the following command:

```bash
python main.py +_runs/test=debug
```

You can similarly run experiments on a cluster using Slurm by choosing an appropriate `platform`, e.g.,

```bash
python main.py +_runs/test=debug +platform=default_platform
```

When running the same experiment multiple times, e.g., for different hyperparameters or changes in the code, you can
use the `_version` flag (defaults to 1) to specify a version number.
This will append a f"v{version}" to the experiment name in the wandb logging.
We similarly have an `idx` parameter (defaults to 1000) that can be used to specify a unique identifier for the
experiment. We use this to identify semantic groupings of experiments ("Baseline X with feature Y") across tasks


## Model checkpointing and loading

Each experiment will be logged in `outputs/hydra/training/${YYYY-MM-DD}/${exp_name}/${run_name}/{$seed}`.
The logs contain folders
* `wandb` for the Weights and Biases logging, which includes metrics and figures
* `checkpoints` for the model checkpoints

To load a model from a checkpoint for evaluation, you can use the `evaluation.py` script, i.e.,
```
python evaluation.py +_evaluations=debug
```

This will load the algorithm config corresponding to that of the *checkpoint* specified in `_test/debug.yaml`, and
then run the evaluation with task settings as described in the test config. This allows loading a given algorithm and
evaluating it on novel task setups.
The results are written to disk as a `.json` file in `outputs/hydra/evaluation/${exp_name}/${run_name}/{$seed}`.
Additionally, plots for the testing data are saved in the same directory.
