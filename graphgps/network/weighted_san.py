import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network
from graphgps.layer.weighted_san_layer import WeightedSANLayer


@register_network("WeightedSANTransformer")
class WeightedSANTransformer(torch.nn.Module):
    """Spectral Attention Network (SAN) Graph Transformer.
    https://arxiv.org/abs/2106.03893

    modified to handle weighted edges (for attack)
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, "The inner and hidden dims must match."

        layers = []
        for _ in range(cfg.gt.layers):
            layers.append(
                WeightedSANLayer(
                    gamma=cfg.gt.gamma,
                    in_dim=cfg.gt.dim_hidden,
                    out_dim=cfg.gt.dim_hidden,
                    num_heads=cfg.gt.n_heads,
                    full_graph=cfg.gt.full_graph,
                    dropout=cfg.gt.dropout,
                    layer_norm=cfg.gt.layer_norm,
                    batch_norm=cfg.gt.batch_norm,
                    residual=cfg.gt.residual,
                    use_bias=cfg.gt.attn.use_bias,
                    gamma_learnable=cfg.gt.gamma_learnable,
                    attn_clamp=cfg.gt.attn.clamp,
                )
            )
        self.layers = torch.nn.Sequential(*layers)
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
