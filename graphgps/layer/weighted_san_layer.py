import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax, scatter


def attn_score_add_biases_and_clamp(score, batch, clamp, fake: bool):
    # disable clamp during attack to let gradient flow
    if (clamp is not None) and (not batch.get("attack_mode", False)):
        score = torch.clamp(score, min=-clamp, max=clamp)

    # for attack cont. relax. bias by log edge probability
    # (edges that are partially fake will be in both scores)
    if fake:
        edge_index = batch.edge_index_fake
        if batch.get("edge_logprob_fake") is not None:
            score = score + batch.edge_logprob_fake.view(-1, 1, 1)
    else:
        edge_index = batch.edge_index
        if batch.get("edge_logprob") is not None:
            score = score + batch.edge_logprob.view(-1, 1, 1)

    # (attack) add the node log probability as bias on the attention
    if batch.get("node_logprob", None) is not None:
        score = score + batch.node_logprob[edge_index[0]].view(-1, 1, 1)
    elif batch.get("node_prob", None) is not None:
        l = torch.zeros_like(batch.node_prob) - torch.inf
        min_prob_mask = batch.node_prob > 1e-30
        l[min_prob_mask] = batch.node_prob[min_prob_mask].log()
        score = score + l[edge_index[0]].view(-1, 1, 1)

    return score



class WeightedMultiHeadAttentionLayer(nn.Module):
    """Multi-Head Graph Attention Layer.

    Ported to PyG and modified compared to the original repo:
    https://github.com/DevinKreuzer/SAN/blob/main/layers/graph_transformer_layer.py
    """

    def __init__(
        self,
        gamma: float,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        full_graph: bool,
        use_bias: bool,
        gamma_learnable: bool,
        clamp: None | float,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        if gamma_learnable:
            self.gamma = nn.Parameter(torch.tensor(0.5, dtype=float), requires_grad=True)
        else:
            self.gamma = gamma
        self.full_graph = full_graph
        self.clamp = abs(clamp) if clamp is not None else None

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

        if self.full_graph:
            self.Q_fake = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
            self.K_fake = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
            self.E_fake = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.d_sqrt = np.sqrt(self.out_dim)

    def propagate_attention(self, batch):
        # (num real edges) x num_heads x out_dim
        score = batch.K_h[batch.edge_index[0]] * batch.Q_h[batch.edge_index[1]]
        # Use available edge features to modify the scores for edges
        if batch.get("E") is not None:
            # E: 1 x num_heads x out_dim and will be broadcast over dim=0
            score = score * batch.E  # (num real edges) x num_heads x out_dim

        # Sum in d and scale by sqrt(d)
        score = score.sum(-1, keepdim=True) / self.d_sqrt
        score = attn_score_add_biases_and_clamp(score, batch, self.clamp, fake=False)
        score = softmax(score, batch.edge_index[1])  # (num real edges) x num_heads x 1

        if self.full_graph:
            # same for fake edges
            score_fake = batch.K_h_fake[batch.edge_index_fake[0]] * batch.Q_h_fake[batch.edge_index_fake[1]]
            if batch.get("E_fake") is not None:
                score_fake = score_fake * batch.E_fake

            score_fake = score_fake.sum(-1, keepdim=True) / self.d_sqrt
            score_fake = attn_score_add_biases_and_clamp(score_fake, batch, self.clamp, fake=True)
            score_fake = softmax(score_fake, batch.edge_index_fake[1])  # (num fake edges) x num_heads x 1

            # scaling by gamma
            score = score / (self.gamma + 1)
            score_fake = self.gamma * score_fake / (self.gamma + 1)

        # Apply attention score to each source node to create edge messages
        msg = batch.V_h[batch.edge_index[0]] * score  # (num real edges) x num_heads x out_dim
        # Add-up real msgs in destination nodes as given by batch.edge_index[1]
        batch.wV = scatter(msg, batch.edge_index[1], dim=0, dim_size=batch.num_nodes, reduce="sum")  # (num nodes in batch) x num_heads x out_dim

        if self.full_graph:
            # Attention via fictional edges
            msg_fake = batch.V_h[batch.edge_index_fake[0]] * score_fake
            # Add messages along fake edges to destination nodes
            batch.wV += scatter(msg_fake, batch.edge_index_fake[1], dim=0, dim_size=batch.num_nodes, reduce="sum")


    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)

        if batch.get("edge_attr_dummy") is not None:
            # one dummy embedding for all real edges; shape: 1 x emb_dim
            E = self.E(batch.edge_attr_dummy)
        else:
            E = None

        if self.full_graph:
            Q_h_fake = self.Q_fake(batch.x)
            K_h_fake = self.K_fake(batch.x)
            if batch.get("edge_attr_dummy_fake") is not None:
                # One dummy embedding used for all fake edges; shape: 1 x emb_dim
                E_fake = self.E_fake(batch.edge_attr_dummy_fake)
            else:
                E_fake = None

        V_h = self.V(batch.x)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        if E is not None:
            batch.E = E.view(-1, self.num_heads, self.out_dim)

        if self.full_graph:
            batch.Q_h_fake = Q_h_fake.view(-1, self.num_heads, self.out_dim)
            batch.K_h_fake = K_h_fake.view(-1, self.num_heads, self.out_dim)
            if E_fake is not None:
                batch.E_fake = E_fake.view(-1, self.num_heads, self.out_dim)

        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(batch)
        h_out = batch.wV
        return h_out


class WeightedSANLayer(nn.Module):
    """Modified GraphTransformerLayer from SAN.

    Ported to PyG from original repo:
    https://github.com/DevinKreuzer/SAN/blob/main/layers/graph_transformer_layer.py
    """

    def __init__(
        self,
        gamma: float,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        full_graph: bool,
        dropout: float,
        layer_norm: bool,
        batch_norm: bool,
        residual: bool,
        use_bias: bool,
        gamma_learnable: bool,
        attn_clamp: None | float,
    ):
        super().__init__()
        assert (out_dim // num_heads) * num_heads == out_dim, "out_dim must be divisible by num_heads"
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.attention = WeightedMultiHeadAttentionLayer(
            gamma=gamma,
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            full_graph=full_graph,
            use_bias=use_bias,
            gamma_learnable=gamma_learnable,
            clamp=attn_clamp,
        )

        self.O_h = nn.Linear(out_dim, out_dim)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        # multi-head attention out
        h_attn_out = self.attention(batch)

        # Concat multi-head outputs
        h = h_attn_out.view(-1, self.out_channels)

        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h  # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        return batch

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.num_heads,
            self.residual,
        )
