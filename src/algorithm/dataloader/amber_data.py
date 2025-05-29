from dataclasses import dataclass
from typing import List, Literal

import torch
from torch_geometric.data.data import Data

from src.algorithm.dataloader.mesh_generation_data import MeshGenerationData
from src.mesh_util.sizing_field_util import get_sizing_field
from src.tasks.domains.mesh_wrapper import MeshWrapper


@dataclass
class AmberData(MeshGenerationData):
    edge_feature_names: List[str] = None
    add_self_edges: bool = True
    initial_mesh_handling: Literal["exclude", "topology_only", "full"] = "exclude"
    refinement_depth: int = 0  # How many times this data has been refined. 0 means this is an initial data point
    sampled_count: int = 0  # how often this piece of data has been sampled.

    def __post_init__(self):
        super().__post_init__()
        if self.initial_mesh_handling not in {"exclude", "topology_only", "full"}:
            raise ValueError(f"Invalid initial_mesh_handling: {self.initial_mesh_handling}. " "Must be one of 'exclude', 'topology_only', or 'full'.")

    @classmethod
    def from_reference(cls, reference: "AmberData", new_mesh: MeshWrapper) -> "AmberData":
        return cls(
            mesh=new_mesh,
            source_data=reference.source_data,
            node_type=reference.node_type,
            sizing_field_interpolation_type=reference.sizing_field_interpolation_type,
            node_feature_names=reference.node_feature_names,
            edge_feature_names=reference.edge_feature_names,
            add_self_edges=reference.add_self_edges,
            initial_mesh_handling=reference.initial_mesh_handling,
            refinement_depth=reference.refinement_depth + 1,
            sampled_count=reference.sampled_count,  # Initialize with the same sampled count as reference mesh
        )

    def increment_sampled_count(self):
        self.sampled_count += 1

    @property
    def observation(self) -> Data:
        if self._observation is None:
            graph = self._get_observation_graph()
            graph.y = torch.tensor(self._labels, dtype=torch.float32)
            self._observation = graph
        return self._observation

    @observation.setter
    def observation(self, value) -> None:
        self._observation = value

    @property
    def graph_size(self) -> int:
        return self.observation.num_nodes + self.observation.num_edges

    def to(self, device) -> "AmberData":
        self.observation = self.observation.to(device)
        return self

    def _get_observation_graph(self) -> Data:
        graph = self._mesh_to_graph(self.mesh)
        if self.initial_mesh_handling in ["full", "topology_only"]:
            graph = self._extend_to_hierarchical_graph(graph)
        return graph

    def _extend_to_hierarchical_graph(self, graph: Data) -> Data:
        """
        Extends the graph to a heterogenous graph that includes the initial mesh as well.
        If self.initial_mesh_handling is either "full" or "topology_only", the graph is extended to include the initial mesh.
        This means
        * adding another observation belonging to the initial mesh. Here, we use all node features that are used for learning
            iff self.initial_mesh_handling == "full", and empty features iff self.initial_mesh_handling == "topology_only".
            We always add the edge features of the initial graph, since they are just relative distances.
        * adding a set of edges between the initial mesh and the learned mesh, to allow for information exchange between
            the two mesh hierarchy levels. Here, for "element", we set an edge iff the midpoint of the learned element
            is inside the initial element. For "node", we set an edge iff the learned node is closest to the initial node.
            The edge features for these edges are the same as for the general observation.
        * adding a one-hot encoding of the node type, i.e., which mesh the node comes from (0: learned, 1: initial) as
            an additional feature to the node features.
        * adding a mask_output attribute to the graph, which is a boolean tensor of shape (num_nodes,) that is True for
            nodes that belong to the learned mesh, and False for nodes that belong to the initial mesh. This is used
            in the GNN forward to mask out the output of the initial mesh nodes, ensuring that the predictions are only
            made on the physical current mesh, instead of the initial mesh.


        Args:
            graph:

        Returns: An extended, hierarchically structured graph over the current and initial mesh.

        """
        if self.refinement_depth == 0:
            # no heterogenous graph, since we directly act on the initial mesh
            graph.x = torch.cat([graph.x, torch.zeros(len(graph.x))[:, None]], dim=1)
            graph.mask_output = torch.ones(len(graph.x)).bool()
        else:  # self.refinement_depth > 0:
            initial_graph = self._mesh_to_graph(self.source_data.initial_mesh)
            from src.mesh_util.transforms.mesh_to_graph import get_inter_graph_edges

            inter_edge_attr, inter_edge_index = get_inter_graph_edges(
                src_mesh=self.mesh,
                dest_mesh=self.source_data.initial_mesh,
                node_type=self.node_type,
                edge_feature_names=self.edge_feature_names,
            )
            graph.edge_index = torch.cat(
                [graph.edge_index, initial_graph.edge_index + len(graph.x), inter_edge_index], dim=1  # offset new graph indices
            )
            graph.edge_attr = torch.cat([graph.edge_attr, initial_graph.edge_attr, inter_edge_attr], dim=0)

            # add one-hot encoding of the node type, i.e., which mesh it comes from (0: learned, 1: initial)
            graph.mask_output = torch.cat([torch.ones(len(graph.x)), torch.zeros(len(initial_graph.x))], dim=0).bool()
            graph.x = torch.cat([graph.x, torch.zeros(len(graph.x))[:, None]], dim=1)

            if self.initial_mesh_handling == "topology_only":
                # mask initial mesh node features
                initial_graph.x = torch.zeros_like(initial_graph.x)
            initial_graph.x = torch.cat([initial_graph.x, torch.ones(len(initial_graph.x))[:, None]], dim=1)
            graph.x = torch.cat([graph.x, initial_graph.x], dim=0)
        return graph

    def _mesh_to_graph(self, mesh: MeshWrapper) -> Data:
        from src.mesh_util.transforms.mesh_to_graph import mesh_to_graph

        graph = mesh_to_graph(
            wrapped_mesh=mesh,
            node_feature_names=self.node_feature_names,
            node_type=self.node_type,
            edge_feature_names=self.edge_feature_names,
            feature_provider=self.feature_provider,
            add_self_edges=self.add_self_edges,
        )
        # add current sizing field as a graph attribute (not as a feature) to use as a baseline for residual prediction
        graph = self._add_current_sizing_field(mesh=mesh, graph=graph)
        return graph

    def _add_current_sizing_field(self, mesh: MeshWrapper, graph: Data) -> Data:
        """
        Adds the current sizing field to the graph as an additional attribute to enable residual learning.
        Args:
            graph:

        Returns:

        """
        sizing_field = get_sizing_field(mesh, mesh_node_type=self.node_type)
        graph.current_sizing_field = torch.Tensor(sizing_field).float()
        return graph
