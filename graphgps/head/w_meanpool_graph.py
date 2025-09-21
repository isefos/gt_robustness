import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import MLP, new_layer_config
from torch_geometric.graphgym.register import register_head
from torch_geometric.nn import global_add_pool, global_mean_pool


@register_head('weighted_mean_pool_graph')
class WeightedMeanPoolGraphHead(torch.nn.Module):
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
            node_prob = batch.node_logprob.exp()
            graph_emb = global_add_pool(batch.x * node_prob[:, None], batch.batch)
            graph_emb /= node_prob.sum().detach()

        elif hasattr(batch, "node_prob") and batch.node_prob is not None:
            graph_emb = global_add_pool(batch.x * batch.node_prob[:, None], batch.batch)
            graph_emb /= batch.node_prob.sum().detach()

        else:
            graph_emb = global_mean_pool(batch.x, batch.batch)

        graph_emb = self.layer_post_mp(graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label
