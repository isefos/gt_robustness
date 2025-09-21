import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer


@register_layer('gcnconvweighted')
class WeightedGCNConvGraphGymLayer(nn.Module):
    """
    Graph Convolutional Network (GCN) layer
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        # cfg.gnn.aggr is not used, aggregation is always 'add', but the normalized adjacency matrix is used, 
        # so the messages are already normalized by the node degrees so 'mean' would be double normalization
        self.model = pyg_nn.GCNConv(
            layer_config.dim_in, layer_config.dim_out, bias=layer_config.has_bias,
        )

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)
        return batch