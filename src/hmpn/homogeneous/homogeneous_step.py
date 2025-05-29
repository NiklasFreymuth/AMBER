from typing import Dict, Any

from torch import nn
from torch_geometric.data.batch import Batch

from src.hmpn.abstract.abstract_step import AbstractStep
from src.hmpn.common.hmpn_util import noop
from src.hmpn.homogeneous.homogeneous_modules import HomogeneousEdgeModule


class HomogeneousStep(AbstractStep):
    """
         Defines a single MessagePassingLayer that takes a homogeneous observation graph and updates its node and edge
         features using different modules (Edge, Node).
         It first updates the edge-features. The node-features are updated next using the new edge-features.
    """

    def __init__(self,
                 stack_config: Dict[str, Any],
                 latent_dimension: int,
                 scatter_reducers,
                 flip_edges_for_nodes: bool = False):
        """
        Initializes the HomogeneousStep, which realizes a single iteration of message passing for a homogeneous graph.
        This message passing layer consists of three modules: Edge, Node, each of which updates the respective
        part of the graph.
        Initializes the HomogeneousStep.
        Args:
            stack_config:
                Configuration of the stack of GNN steps. Should contain keys
                "num_steps" (int),
                "residual_connections" (str: "none", "inner", "outer"),
                "mlp" (Dict[str, Any]). "mlp" is a dictionary for the general configuration of the MLP.
                    which should contain keys
                    "num_layers" (int),
                    "add_output_layer" (bool),
                    "activation_function" (str: "relu", "leakyrelu", "tanh", "silu"), and
                    "regularization" (Dict[str, Any]),
                        which should contain keys
                        "spectral_norm" (bool),
                        "dropout" (float),
                        "latent_normalization" (str: "batch_norm", "layer_norm" or None)
            latent_dimension:
                Dimension of the latent space.
            scatter_reducers:
                reduce operators from torch_scatter. Can be e.g. [scatter_mean]
        """

        super().__init__(stack_config=stack_config,
                         latent_dimension=latent_dimension)
        mlp_config = stack_config["mlp"]


        # edge module
        self.edge_module = HomogeneousEdgeModule(latent_dimension=latent_dimension,
                                                 mlp_config=mlp_config,
                                                 scatter_reducers=scatter_reducers)

        node_update_type = stack_config.get("node_update_type", "message_passing")
        if node_update_type == "message_passing":
            from src.hmpn.homogeneous.homogeneous_modules import HomogeneousMessagePassingNodeModule
            self.node_module = HomogeneousMessagePassingNodeModule(latent_dimension=latent_dimension,
                                                                   mlp_config=mlp_config,
                                                                   scatter_reducers=scatter_reducers,
                                                                   flip_edges_for_nodes=flip_edges_for_nodes)
        elif node_update_type == "gat":
            from src.hmpn.homogeneous.homogeneous_modules import HomogeneousGatNodeModule
            self.node_module = HomogeneousGatNodeModule(latent_dimension=latent_dimension,
                                                        flip_edges_for_nodes=flip_edges_for_nodes,
                                                        heads=stack_config.get("attention_heads", 4))

        self.reset_parameters()

        if self.use_layer_norm:
            self._node_layer_norms = nn.LayerNorm(normalized_shape=latent_dimension)
            self._edge_layer_norms = nn.LayerNorm(normalized_shape=latent_dimension)
        else:
            self._node_layer_norms = None
            self._edge_layer_norms = None

    def _store_nodes(self, graph: Batch):
        self._old_graph["x"] = graph.x

    def _store_edges(self, graph: Batch):
        self._old_graph["edge_attr"] = graph.edge_attr

    def _add_node_residual(self, graph: Batch):
        graph.__setattr__("x", graph.x + self._old_graph["x"])

    def _add_edge_residual(self, graph: Batch):
        graph.__setattr__("edge_attr", graph.edge_attr + self._old_graph["edge_attr"])

    def _node_layer_norm(self, graph: Batch) -> None:
        graph.__setattr__("x", self._node_layer_norms(graph.x))

    def _edge_layer_norm(self, graph: Batch) -> None:
        graph.__setattr__("edge_attr", self._edge_layer_norms(graph.edge_attr))
