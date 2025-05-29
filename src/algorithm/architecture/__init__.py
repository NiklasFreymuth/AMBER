import torch
from omegaconf import DictConfig
from torch import nn
from torch_geometric.data import Data

from src.mesh_util.transforms.mesh_to_image import MeshImage


def get_gnn(architecture_config: DictConfig, example_graph: Data) -> nn.Module:
    if architecture_config.name == "mpn":
        from src.algorithm.architecture.supervised_mpn import SupervisedMPN

        return SupervisedMPN(architecture_config=architecture_config, example_graph=example_graph)
    elif architecture_config.name == "graphmesh_gcn":
        from src.algorithm.architecture.graphmesh_gcn import GraphmeshGCN

        return GraphmeshGCN(architecture_config=architecture_config, example_graph=example_graph)
    else:
        raise ValueError(f"Unknown GNN architecture {architecture_config.name}")


def get_cnn(architecture_config: DictConfig, example_mesh_image: MeshImage) -> nn.Module:
    if architecture_config.name == "unet":
        from src.algorithm.architecture.unet import UNet

        example_image = example_mesh_image.features
        out_features = 1 if example_mesh_image.labels.ndim == 1 else example_mesh_image.labels.shape[1]
        return UNet(architecture_config=architecture_config, example_image=example_image, out_features=out_features)
    else:
        raise ValueError(f"Unknown CNN architecture {architecture_config.name}")
