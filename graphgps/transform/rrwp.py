import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from torch_geometric.graphgym.config import cfg


def add_full_rrwp(data: Data, walk_length: int):
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    device = edge_index.device
    edge_weight_rrwp = data.edge_weight

    if not cfg.attack.GRIT.cont_RRWP:
        edge_weight_rrwp = None
    elif not cfg.attack.GRIT.grad_RRWP and edge_weight_rrwp is not None:
        edge_weight_rrwp = edge_weight_rrwp.detach()

    adj = SparseTensor.from_edge_index(
        edge_index,
        edge_weight_rrwp,
        sparse_sizes=(num_nodes, num_nodes),
    )

    # Compute D^{-1} A:
    deg = adj.sum(dim=1)
    deg_mask = deg != 0
    deg_inv = torch.zeros_like(deg)
    deg_inv[deg_mask] = 1.0 / deg[deg_mask]
    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()

    pe_list = [
        torch.eye(num_nodes, dtype=torch.float, device=device),
        adj,
    ]

    out = adj
    for _ in range(walk_length - 2):
        out = out @ adj
        pe_list.append(out)

    pe = torch.stack(pe_list, dim=-1) # n x n x k

    abs_pe = pe.diagonal().transpose(0, 1) # n x k

    rel_pe = SparseTensor.from_dense(pe, has_value=True)
    rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
    rel_pe_idx = torch.stack([rel_pe_row, rel_pe_col], dim=0)

    data.rrwp = abs_pe
    data.rrwp_index = rel_pe_idx
    data.rrwp_val = rel_pe_val

    # recompute degrees, because they may have different cont. / grad. settings than the rrwp
    if data.edge_weight is None or (not cfg.attack.GRIT.cont_degree):
        degrees = torch.zeros(num_nodes, dtype=torch.long, device=device).scatter_(
            value=1,
            index=edge_index[1, :],
            dim=0,
            reduce="add",
        )
    else:
        edge_weight_deg = data.edge_attr
        if not cfg.attack.GRIT.grad_degree:
            edge_weight_deg = edge_weight_deg.detach()
        degrees = torch.scatter_add(
            input=torch.zeros(num_nodes, device=device),
            src=edge_weight_deg,
            index=edge_index[1, :],
            dim=0,
        )

    data.log_deg = torch.log1p(degrees)

    return data
