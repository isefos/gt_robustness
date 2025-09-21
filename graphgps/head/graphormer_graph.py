import torch
from torch_geometric.graphgym.models.layer import MLP, new_layer_config
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head


@register_head('graphormer_graph')
class GraphormerHead(torch.nn.Module):
    """
    Graphormer prediction head for graph prediction tasks.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]

        self.ln = torch.nn.LayerNorm(dim_in)
        self.layer_post_mp = MLP(
            new_layer_config(
                dim_in, dim_out, cfg.gnn.layers_post_mp, has_act=False, has_bias=True, cfg=cfg,
            )
        )

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        x = self.ln(batch.x)
        graph_emb = self.pooling_fun(x, batch.batch)
        graph_emb = self.layer_post_mp(graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label
