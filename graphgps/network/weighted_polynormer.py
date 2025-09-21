import torch
import torch.nn.functional as F
from graphgps.layer.weighted_polynormer_global import WeightedPolynormerGlobal
from graphgps.layer.weighted_polynormer_local import WeightedPolynormerLocal
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP


@register_network("WeightedPolynormer")
class Polynormer(torch.nn.Module):
    """
    Adapted from https://github.com/cornell-zhang/Polynormer/blob/master/model.py
    """

    def __init__(self, dim_in: int, dim_out: int):
        super(Polynormer, self).__init__()
        # flag for local only pre-training
        self._global = False

        # "pre-model" steps
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in
        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner
        assert cfg.gnn.dim_inner == dim_in, "The inner and hidden dims must match."

        self.in_drop = cfg.gt.polynormer.dropout_node_input  # 0.15

        # local
        self.local_layers = torch.nn.ModuleList()
        for _ in range(cfg.gnn.layers_mp):
            self.local_layers.append(
                WeightedPolynormerLocal(
                    heads=cfg.gnn.att_heads,
                    dim_hidden=dim_in,
                    beta=cfg.gt.polynormer.beta,
                    dropout=cfg.gnn.dropout,
                    pre_ln=cfg.gt.polynormer.local_pre_layer_norm,
                    residual=False,  # original Polynormer residual implemented here between layers
                )
            )
        
        # global
        self.global_layers = torch.nn.ModuleList()
        for _ in range(cfg.gt.layers):
            self.global_layers.append(
                WeightedPolynormerGlobal(
                    dim_hidden=dim_in,
                    heads=cfg.gt.n_heads,
                    beta=cfg.gt.polynormer.beta,
                    dropout=cfg.gt.dropout,
                    qk_shared=cfg.gt.polynormer.qk_shared,  # True
                    residual=False,  # in original Polynormer implementation
                )
            )

        # head
        assert dim_in == cfg.gnn.dim_inner
        GNNHead = register.head_dict[cfg.gnn.head]
        # is used in local pre-training, then "discarded" (global used instead)
        self.post_gnn_local = GNNHead(dim_in=dim_in, dim_out=dim_out)
        self.post_gnn_global = GNNHead(dim_in=dim_in, dim_out=dim_out)

    def reset_parameters(self):
        for local_l in self.local_layers:
            local_l.reset_parameters()
        self.global_attn.reset_parameters()
        self.post_gnn_local.reset_parameters()
        self.post_gnn_global.reset_parameters()
        if self.beta < 0:
            torch.nn.init.xavier_normal_(self.betas)
        else:
            torch.nn.init.constant_(self.betas, self.beta)

    def forward(self, batch):
        batch = self.encoder(batch)
        if cfg.gnn.layers_pre_mp > 0:
            batch = self.pre_mp(batch)

        # TODO: remove when general input dropout is implemented (to avoid double dropout)
        batch.x = F.dropout(batch.x, p=self.in_drop, training=self.training)

        # using the original Polynormer residual connection (sum of all layer outputs)
        x_local = 0
        for local_layer in self.local_layers:
            batch = local_layer(batch)
            x_local = x_local + batch.x
        batch.x = x_local

        if self._global:
            # original implementation does a layer norm here first,
            # but first op in global is also layer norm, so not needed
            for global_layer in self.global_layers:
                batch = global_layer(batch)
            batch = self.post_gnn_global(batch)
        else:
            batch = self.post_gnn_local(batch)

        return batch
