import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import MLP, new_layer_config
from torch_geometric.graphgym.register import register_head
from torch_geometric.nn import global_add_pool


@register_head('weighted_add_pool_graph')
class WeightedAddPoolGraphHead(torch.nn.Module):
    """
    GNN prediction head for graph prediction tasks.
    The optional post_mp layer (specified by cfg.gnn.post_mp) is used
    to transform the pooled embedding using an MLP.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.layer_post_mp = MLP(
            new_layer_config(
                dim_in, dim_out, cfg.gnn.layers_post_mp, has_act=False, has_bias=True, cfg=cfg,
            )
        )

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):

        if hasattr(batch, "node_logprob") and batch.node_logprob is not None:
            graph_emb = global_add_pool(batch.x * batch.node_logprob.exp()[:, None], batch.batch)

        elif hasattr(batch, "node_prob") and batch.node_prob is not None:
            graph_emb = global_add_pool(batch.x * batch.node_prob[:, None], batch.batch)

        else:
            graph_emb = global_add_pool(batch.x, batch.batch)
            
        graph_emb = self.layer_post_mp(graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label
