"""
based on Polynormer implementation from https://github.com/cornell-zhang/Polynormer/blob/master/model.py
added:
 - batched computation
 - continuous relaxations for attack
"""
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch


class WeightedPolynormerGlobal(torch.nn.Module):
    def __init__(
        self,
        dim_hidden: int,
        heads: int,
        beta: float,
        dropout: float,  # TODO: add attention dropout implementation
        qk_shared: bool,
        residual: bool,
    ):
        super(WeightedPolynormerGlobal, self).__init__()
        self.heads = heads
        self.beta = beta
        self.dropout = dropout
        self.qk_shared = qk_shared
        self.residual = residual

        self.dim_per_head = dim_hidden // heads
        assert self.dim_per_head * heads == dim_hidden, (
            f"`dim_hidden={dim_hidden}` must be divisible by `heads={heads}`"
        )

        if self.beta < 0:
            self.b = torch.nn.Parameter(torch.zeros(dim_hidden))
        else:
            self.b = torch.nn.Parameter(torch.ones(dim_hidden) * self.beta)

        self.h_lin = torch.nn.Linear(dim_hidden, dim_hidden)
        if not self.qk_shared:
            self.q_lin = torch.nn.Linear(dim_hidden, dim_hidden)
        self.k_lin = torch.nn.Linear(dim_hidden, dim_hidden)
        self.v_lin = torch.nn.Linear(dim_hidden, dim_hidden)

        # TODO: allow for using batch norm?
        self.input_norm = torch.nn.LayerNorm(dim_hidden)
        self.norm = torch.nn.LayerNorm(dim_hidden)

        self.lin_out = torch.nn.Linear(dim_hidden, dim_hidden)

    def reset_parameters(self):
        self.h_lin.reset_parameters()
        if not self.qk_shared:
            self.q_lin.reset_parameters()
        self.k_lin.reset_parameters()
        self.v_lin.reset_parameters()
        self.input_norm.reset_parameters()
        self.norm.reset_parameters()
        if self.beta < 0:
            torch.nn.init.xavier_normal_(self.b)
        else:
            torch.nn.init.constant_(self.b, self.beta)
        self.lin_out.reset_parameters()

    def forward(self, batch):
        x = self.input_norm(batch.x)
        x_in = x
        h = self.h_lin(x)

        # TODO: dense batch only after qkv projections more efficient?

        # dense batching and dimensions
        x, mask = to_dense_batch(x, batch.batch)
        B, N_max, D = x.shape  # B is num graphs
        D_h = self.dim_per_head
        H = self.heads

        # keys and queries with sigmoid to ensure positive and limit range
        k = F.sigmoid(self.k_lin(x))
        if self.qk_shared:
            q = k
        else:
            q = F.sigmoid(self.q_lin(x))
        # values
        v: torch.Tensor = self.v_lin(x)  # (B, N_max, D)

        # attention masking (to expanded fake nodes of dense batching) / node prob weighting
        if hasattr(batch, "node_logprob"):
            node_prob, _ = to_dense_batch(batch.node_logprob.exp(), batch.batch)
        elif hasattr(batch, "node_prob"):
            node_prob, _ = to_dense_batch(batch.node_prob, batch.batch)
        else:
            # all real nodes get prob = 1
            node_prob = mask.float()  # (B, N_max)

        ##### alternative implementation:
        # numerator
        #kv = torch.einsum('bnih, bnjh, bn -> bijh', k, v, node_prob)  # (B, D_h, D_h, H)
        #num = torch.einsum('bidh, bdjh -> bijh', q, kv)  # (B, N_max, D_h, H)
        # denominator
        #k_sum = torch.einsum('bndh, bn -> bdh', k, node_prob)  # (B, D_h, H)
        #den = torch.einsum('bndh, bdh -> bnh', q, k_sum).unsqueeze(2)  # (B, N_max, 1, H)

        # linear global attention output, concatenate head outputs
        #x = (num / den).reshape(B, N_max, -1)  # (B, N_max, D)
        #####

        k = k * node_prob[:, :, None]
        # TODO: implement attention dropout (not used in original paper)

        # reshape the tensors
        q = q.view(B, N_max, H, D_h).transpose(1, 2).reshape(B*H, N_max, D_h)
        v = v.view(B, N_max, H, D_h).transpose(1, 2).reshape(B*H, N_max, D_h)
        k = k.transpose(1, 2).reshape(B*H, D_h, N_max)

        # numerator
        kv = torch.bmm(k, v)  # (B*H, D_h, D_h)
        num = torch.bmm(q, kv)  # (B*H, N_max, D_h)
        # denominator
        k_sum = k.sum(-1, keepdim=True)  # (B*H, D_h, 1)
        den = torch.bmm(q, k_sum)  # (B*H, N_max, 1)

        # linear global attention output, concatenate head outputs
        x = (num / den)  # (B*H, N_max, D_h)
        x = x.view(B, H, N_max, D_h).transpose(1, 2).reshape(B, N_max, D)  # (B, N_max, D)
        # undo dense batching
        x = x[mask, :]  # (N_total, D)

        # 2nd order polynomial update
        if self.beta < 0:
            beta = F.sigmoid(self.b).unsqueeze(0)  # (1, D)
        else:
            beta = self.b.unsqueeze(0)
        x = self.norm(x) * (h + beta)
        x = F.relu(self.lin_out(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.residual:
            x = x + x_in
        batch.x = x
        return batch
