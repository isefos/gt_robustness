"""
based on Polynormer implementation from https://github.com/cornell-zhang/Polynormer/blob/master/model.py
added:
 - continuous relaxations for attack
"""
import torch
import torch.nn.functional as F
from graphgps.layer.weighted_gat_layer import WeightedGATConvLayer


class WeightedPolynormerLocal(torch.nn.Module):
    def __init__(
        self,
        heads: int,
        dim_hidden: int,
        beta: float,
        dropout: float,
        pre_ln: bool,
        residual: bool,
    ):
        super(WeightedPolynormerLocal, self).__init__()
        self.dropout = dropout
        self.pre_ln = pre_ln
        self.residual = residual

        self.beta = beta
        if self.beta < 0:
            self.b = torch.nn.Parameter(torch.zeros(dim_hidden))
        else:
            self.b = torch.nn.Parameter(torch.ones(dim_hidden) * self.beta)

        dim_per_head = dim_hidden // heads
        assert dim_per_head * heads == dim_hidden, (
            f"`dim_hidden={dim_hidden}` must be divisible by `heads={heads}`"
        )

        self.h_lin = torch.nn.Linear(dim_hidden, dim_hidden)
        self.local_conv = WeightedGATConvLayer(
            dim_hidden,
            dim_per_head,
            heads=heads,
            concat=True,
            add_self_loops=False,
            bias=False,
        )
        self.lin = torch.nn.Linear(dim_hidden, dim_hidden)
        # TODO: make possible to choose batch norm
        self.norm = torch.nn.LayerNorm(dim_hidden)
        if self.pre_ln:
            self.input_norm = torch.nn.LayerNorm(dim_hidden)

    def reset_parameters(self):
        self.local_conv.reset_parameters()
        self.lin.reset_parameters()
        self.h_lin.reset_parameters()
        self.norm.reset_parameters()
        if self.pre_ln:
            self.input_norm.reset_parameters()
        if self.beta < 0:
            torch.nn.init.xavier_normal_(self.b)
        else:
            torch.nn.init.constant_(self.b, self.beta)

    def forward(self, batch):
        x = batch.x
        if self.pre_ln:
            x = self.input_norm(x)
        x_in = x

        h = self.h_lin(x)
        h = F.relu(h)

        x = self.local_conv(x, batch.edge_index, batch.edge_attr) + self.lin(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if self.beta < 0:
            beta = F.sigmoid(self.b).unsqueeze(0)
        else:
            beta = self.b.unsqueeze(0)
        x = ((1 - beta) * self.norm(h * x)) + (beta * x)

        if self.residual:
            x = x + x_in
        batch.x = x
        return batch
