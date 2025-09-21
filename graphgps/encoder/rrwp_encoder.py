import torch
from torch import nn
from torch_geometric.graphgym.register import (
    register_edge_encoder,
    register_node_encoder,
)
from torch_geometric.utils import (
    add_self_loops,
    unbatch_edge_index,
    coalesce,
    cumsum,
)
from torch_geometric.graphgym.config import cfg


def complementary_edge_index(edge_index, batch=None):
    """
    Returns batched sparse adjacency matrices with exactly those edges that
    are not in the input `edge_index`.
    Args:
        edge_index: The edge indices.
        batch: Batch vector, which assigns each node to a specific example.
    Returns:
        Complementary edge index.
    """

    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    num_nodes = batch.new_zeros(batch_size).scatter_(
        value=1,
        index=batch,
        dim=0,
        reduce="add",
    )
    cum_nodes = cumsum(num_nodes)
    edge_indices = unbatch_edge_index(edge_index, batch)

    c_index_list = []
    for i in range(batch_size):
        n = num_nodes[i].item()
        e = edge_indices[i]
        adj = torch.ones((n, n), dtype=torch.short, device=edge_index.device)
        adj[e[0], e[1]] = 0
        c_edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        c_index_list.append(c_edge_index + cum_nodes[i])

    c_edge_index = torch.cat(c_index_list, dim=1).contiguous()    
    return c_edge_index


@register_node_encoder("RRWPLinearNode")
class RRWPLinearNodeEncoder(torch.nn.Module):
    """
    FC_1(RRWP) + FC_2(Node-attr)
    note: FC_2 is given by the Typedict encoder of node-attr in some cases
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.fc = nn.Linear(cfg.posenc_RRWP.ksteps, emb_dim, bias=False)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, batch):
        rrwp = self.fc(batch.rrwp)
        if "x" in batch:
            batch.x = batch.x + rrwp
        else:
            batch.x = rrwp
        return batch


@register_edge_encoder("RRWPLinearEdge")
class RRWPLinearEdgeEncoder(torch.nn.Module):
    """
    Merge RRWP with given edge-attr and zero-padding to all pairs of node
    FC_1(RRWP) + FC_2(edge-attr)
    - FC_2 given by the TypedictEncoder in same cases
    - Zero-padding for non-existing edges in fully-connected graph
    - (optional) add node-attr as the E_{i,i}'s attr
        note: assuming  node-attr and edge-attr have same dimension after Encoders
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.fc = nn.Linear(cfg.posenc_RRWP.ksteps, emb_dim, bias=False)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.pad_to_full_graph = cfg.gt.attn.full_attn
        padding = torch.zeros(1, emb_dim, dtype=torch.float)
        self.register_buffer("padding", padding)

    def forward(self, batch):
        rrwp_idx = batch.rrwp_index
        rrwp_val = self.fc(batch.rrwp_val)

        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        if edge_attr is None:
            edge_attr = edge_index.new_zeros(edge_index.size(1), rrwp_val.size(1))
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=batch.num_nodes, fill_value=0.)

        out_idx, out_val = coalesce(
            torch.cat([edge_index, rrwp_idx], dim=1),
            torch.cat([edge_attr, rrwp_val], dim=0),
            batch.num_nodes,
            reduce="sum",
        )

        if self.pad_to_full_graph:
            c_edge_index = complementary_edge_index(out_idx, batch=batch.batch)
            edge_attr_pad = self.padding.repeat(c_edge_index.size(1), 1)
            # zero padding to fully-connected graphs
            out_idx = torch.cat([out_idx, c_edge_index], dim=1)
            out_val = torch.cat([out_val, edge_attr_pad], dim=0)
            out_idx, out_val = coalesce(
               out_idx,
               out_val,
               batch.num_nodes,
               reduce="sum",
            )

        batch.edge_index, batch.edge_attr = out_idx, out_val
        return batch

    def __repr__(self):
        r = (
            f"{self.__class__.__name__}"
            f"(pad_to_full_graph={self.pad_to_full_graph},"
            f"{self.fc.__repr__()})"
        )
        return r
