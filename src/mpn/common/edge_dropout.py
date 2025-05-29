from torch import nn
from torch_geometric.data.batch import Batch
from torch_geometric.utils import dropout_edge


class EdgeDropout(nn.Module):
    """
    Small wrapper around torch_geometric dropout_edge to allow for easy use as a torch module
    """

    def __init__(self, dropout_prob: float):
        super(EdgeDropout, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, graph: Batch):
        """
        In-place dropout of edges in the graph.
        Args:
            graph:

        Returns:

        """
        graph.edge_index, edge_mask = dropout_edge(edge_index=graph.edge_index, p=self.dropout_prob, training=self.training)
        graph.edge_attr = graph.edge_attr[edge_mask]

    def __repr__(self):
        return f"EdgeDropout(p={self.dropout_prob})"
