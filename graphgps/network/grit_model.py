import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP, FeatureEncoder
from torch_geometric.graphgym.register import register_network
from graphgps.layer.grit_layer import GritTransformerLayer


@register_network("GritTransformer")
class GritTransformer(torch.nn.Module):
    """
    GritTransformer (Graph Inductive Bias Transformer)
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
                GritTransformerLayer(
                    in_dim=cfg.gt.dim_hidden,
                    out_dim=cfg.gt.dim_hidden,
                    num_heads=cfg.gt.n_heads,
                    dropout=cfg.gt.dropout,
                    attn_dropout=cfg.gt.attn_dropout,
                    layer_norm=cfg.gt.layer_norm,
                    batch_norm=cfg.gt.batch_norm,
                    act=cfg.gnn.act,
                    norm_e=cfg.gt.attn.norm_e,
                    O_e=cfg.gt.attn.O_e,
                    update_e=cfg.gt.update_e,
                    bn_momentum=cfg.gt.bn_momentum,
                    bn_no_runner=cfg.gt.bn_no_runner,
                    deg_scaler=cfg.gt.attn.deg_scaler,
                    attn_use_bias=cfg.gt.attn.use_bias,
                    attn_clamp=cfg.gt.attn.clamp,
                    attn_act=cfg.gt.attn.act,
                    attn_edge_enhance=cfg.gt.attn.edge_enhance,
                )
            )
        self.layers = torch.nn.Sequential(*layers)
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
