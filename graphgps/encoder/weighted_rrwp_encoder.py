import torch
from torch import nn
from torch_geometric.graphgym.register import register_node_encoder
from torch_geometric.utils import add_self_loops, coalesce
from graphgps.encoder.rrwp_encoder import complementary_edge_index
from graphgps.transform.rrwp import add_full_rrwp
from torch_geometric.graphgym.config import cfg


@register_node_encoder("WeightedRRWPLinear")
class WeightedRRWPLinearEncoder(torch.nn.Module):
    """
    FC_1_n(RRWP) + FC_2_n(Node-attr) for node
    FC_1_e(RRWP) for edges (+ optionally dummy features for existing edges)
    note: FC_2_n is given by the Typedict encoder of node-attr in some cases
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.pad_to_full_graph = cfg.gt.attn.full_attn
        self.walk_length = cfg.posenc_RRWP.ksteps
        self.add_dummy_edge_attr = cfg.posenc_RRWP.w_add_dummy_edge

        if self.add_dummy_edge_attr:
            self.dummy_edge_encoder = torch.nn.Embedding(num_embeddings=1, embedding_dim=emb_dim)

        self.fc_node = nn.Linear(self.walk_length, emb_dim, bias=False)
        self.fc_edge = nn.Linear(self.walk_length, emb_dim, bias=False)
        torch.nn.init.xavier_uniform_(self.fc_node.weight)
        torch.nn.init.xavier_uniform_(self.fc_edge.weight)

        padding = torch.zeros(1, emb_dim, dtype=torch.float)
        self.register_buffer("padding", padding)

    def forward(self, batch):
        attack_mode = batch.get("attack_mode", False)
        if attack_mode or batch.get("rrwp") is None:
            # for attack
            assert batch.num_graphs == 1, "On the fly preprocessing only works for single graphs"
            batch.edge_weight = batch.edge_attr
            add_full_rrwp(batch, walk_length=self.walk_length)

        # node
        rrwp = self.fc_node(batch.rrwp)
        if "x" in batch:
            batch.x = batch.x + rrwp
        else:
            batch.x = rrwp

        # edge
        rrwp_idx = batch.rrwp_index
        rrwp_val = self.fc_edge(batch.rrwp_val)

        edge_index = batch.edge_index
        if self.add_dummy_edge_attr:
            dummy_attr = edge_index.new_zeros(edge_index.size(1))
            edge_attr = self.dummy_edge_encoder(dummy_attr)
            if attack_mode and batch.edge_weight is not None and cfg.attack.GRIT.dummy_edge_weighting:
                # for attack, weighted dummy features
                edge_attr = edge_attr * batch.edge_weight[:, None]
        else:
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
            f"{self.fc_node.__repr__()},"
            f"{self.fc_edge.__repr__()})"
        )
        return r
