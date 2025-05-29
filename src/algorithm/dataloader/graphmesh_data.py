from dataclasses import dataclass
from typing import List

from torch_geometric.data import Data

from src.algorithm.dataloader.amber_data import AmberData
from src.tasks.domains.mesh_wrapper import MeshWrapper


@dataclass
class GraphMeshData(AmberData):
    boundary_graph_feature_names: List[str] = None

    def __post_init__(self):
        # Ensure some constraints for this baseline compared to the (more general) AMBER.
        super().__post_init__()
        assert self.initial_mesh_handling == "exclude"
        assert self.refinement_depth == 0
        assert self.sizing_field_interpolation_type == "sampled_vertex"

    def _get_observation_graph(self) -> Data:
        graph = super()._get_observation_graph()
        graph.boundary_vertex_graphs = self._get_boundary_vertex_graphs(mesh=self.mesh)
        return graph

    def _get_boundary_vertex_graphs(self, mesh: MeshWrapper) -> List[Data]:
        from src.mesh_util.transforms.mesh_to_boundary_vertex_graph import (
            get_boundary_vertex_graphs,
        )

        boundary_features = get_boundary_vertex_graphs(
            mesh, feature_provider=self.feature_provider, boundary_graph_feature_names=self.boundary_graph_feature_names
        )
        return boundary_features
