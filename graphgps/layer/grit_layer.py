import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax, scatter
import torch_geometric.graphgym.register as register
import opt_einsum as oe


class MultiHeadAttentionLayerGritSparse(nn.Module):
    """
    Attention Computation for GRIT
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        use_bias: bool,
        clamp: None | float,  # = 5.,
        dropout: float,  # = 0.,
        act: None | str,  # = None,
        edge_enhance: bool,  # = True,
    ):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads * 2, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)

        self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1), requires_grad=True)
        nn.init.xavier_normal_(self.Aw)

        if act is None:
            self.act = nn.Identity()
        else:
            self.act = register.act_dict[act]()

        if self.edge_enhance:
            self.VeRow = nn.Parameter(
                torch.zeros(self.out_dim, self.num_heads, self.out_dim),
                requires_grad=True,
            )
            nn.init.xavier_normal_(self.VeRow)

    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]  # (num relative) x num_heads x out_dim
        dest = batch.Q_h[batch.edge_index[1]]
        score = src + dest

        if batch.get("E", None) is not None:
            batch.E = batch.E.view(-1, self.num_heads, self.out_dim * 2)
            E_w, E_b = batch.E[:, :, :self.out_dim], batch.E[:, :, self.out_dim:]
            score = score * E_w
            score = torch.sqrt(torch.relu(score) + 1e-8) - torch.sqrt(torch.relu(-score) + 1e-8)
            score = score + E_b

        score = self.act(score)
        e_t = score

        # output edge
        if batch.get("E", None) is not None:
            batch.wE = score.flatten(1)

        # final attn
        score = oe.contract("ehd, dhc->ehc", score, self.Aw, backend="torch")

        # disable clamp during attack to let gradient flow
        if (self.clamp is not None) and (not batch.get("attack_mode", False)):
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        # for attack cont. relax. add the node log probability as bias on the attention
        if batch.get("node_logprob", None) is not None:
            score = score + batch.node_logprob[batch.edge_index[0]].view(-1, 1, 1)
        elif batch.get("node_prob", None) is not None:
            l = torch.zeros_like(batch.node_prob) - torch.inf
            min_prob_mask = batch.node_prob > 1e-30
            l[min_prob_mask] = batch.node_prob[min_prob_mask].log()
            score = score + l[batch.edge_index[0]].view(-1, 1, 1)
        
        score = softmax(score, batch.edge_index[1])  # (num relative) x num_heads x 1
        score = self.dropout(score)
        batch.attn = score

        # Aggregate with Attn-Score
        msg = batch.V_h[batch.edge_index[0]] * score
        # (num nodes in batch) x num_heads x out_dim
        batch.wV = scatter(msg, batch.edge_index[1], dim=0, dim_size=batch.num_nodes, reduce="sum")

        if self.edge_enhance and batch.E is not None:
            # (num nodes in batch) x num_heads x out_dim
            rowV = scatter(e_t * score, batch.edge_index[1], dim=0, reduce="sum")
            rowV = oe.contract("nhd, dhc -> nhc", rowV, self.VeRow, backend="torch")
            batch.wV = batch.wV + rowV

    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)

        V_h = self.V(batch.x)
        if batch.get("edge_attr", None) is not None:
            batch.E = self.E(batch.edge_attr)
        else:
            batch.E = None

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(batch)
        h_out = batch.wV
        e_out = batch.get('wE', None)

        return h_out, e_out


class GritTransformerLayer(nn.Module):
    """
    Transformer Layer for GRIT
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        dropout: float,  # 0.0
        attn_dropout: float,  # 0.0
        layer_norm: bool,  # False
        batch_norm: bool,  # True
        act: None | str,  # 'relu'
        norm_e: bool,  # True
        O_e: bool,  # True
        update_e: bool,
        bn_momentum: float,
        bn_no_runner: bool,
        deg_scaler: bool,
        attn_use_bias: bool,
        attn_clamp: None | float,
        attn_act: None | str,
        attn_edge_enhance: bool,
    ):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.update_e = update_e
        self.bn_momentum = bn_momentum
        self.bn_no_runner = bn_no_runner
        self.deg_scaler = deg_scaler

        self.act = register.act_dict[act]() if act is not None else nn.Identity()
        assert (out_dim // num_heads) * num_heads == out_dim, "out_dim must be divisible by num_heads"
        self.attention = MultiHeadAttentionLayerGritSparse(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=attn_use_bias,
            clamp=attn_clamp,
            dropout=attn_dropout,
            act=attn_act,
            edge_enhance=attn_edge_enhance,
        )

        self.O_h = nn.Linear(out_dim, out_dim)
        if O_e:
            self.O_e = nn.Linear(out_dim, out_dim)
        else:
            self.O_e = nn.Identity()

        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, out_dim, 2))
            nn.init.xavier_normal_(self.deg_coef)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim) if norm_e else nn.Identity()

        if self.batch_norm:
            # when the batch_size is really small, use smaller momentum to avoid bad mini-batch 
            # leading to extremely bad val/test loss (NaN)
            self.batch_norm1_h = nn.BatchNorm1d(
                out_dim,
                track_running_stats=not self.bn_no_runner,
                eps=1e-5,
                momentum=self.bn_momentum,
            )
            self.batch_norm1_e = nn.BatchNorm1d(
                out_dim,
                track_running_stats=not self.bn_no_runner,
                eps=1e-5,
                momentum=self.bn_momentum,
            ) if norm_e else nn.Identity()

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(
                out_dim,
                track_running_stats=not self.bn_no_runner,
                eps=1e-5,
                momentum=self.bn_momentum,
            )

    def forward(self, batch):
        h = batch.x
        num_nodes = batch.num_nodes
        log_deg = batch.log_deg.view(num_nodes, 1)

        h_in1 = h  # for first residual connection
        e_in1 = batch.get("edge_attr", None)
        e = None
        h_attn_out, e_attn_out = self.attention(batch)

        h = h_attn_out.view(num_nodes, -1)
        h = F.dropout(h, self.dropout, training=self.training)

        if self.deg_scaler:
            h = torch.stack([h, h * log_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h = self.O_h(h)
        if e_attn_out is not None:
            e = e_attn_out.flatten(1)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.O_e(e)

        # residual:
        h = h_in1 + h
        if e is not None:
            e = e + e_in1

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            if e is not None:
                e = self.layer_norm1_e(e)
        if self.batch_norm:
            h = self.batch_norm1_h(h)
            if e is not None:
                e = self.batch_norm1_e(e)

        # FFN for h
        h_in2 = h  # for second residual connection
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        # residual:
        h = h_in2 + h

        if self.layer_norm:
            h = self.layer_norm2_h(h)
        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        if self.update_e:
            batch.edge_attr = e
        else:
            batch.edge_attr = e_in1

        return batch

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={})\n[{}]'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.num_heads,
            super().__repr__(),
        )
