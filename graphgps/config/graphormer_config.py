from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_graphormer')
def set_cfg_gt(cfg):

    # TODO: move to gnn / gt / ... configs to unify the configs as much as possible
    cfg.graphormer = CN()
    cfg.graphormer.num_layers = 6
    cfg.graphormer.embed_dim = 80
    cfg.graphormer.num_heads = 4
    cfg.graphormer.dropout = 0.0
    cfg.graphormer.attention_dropout = 0.0
    cfg.graphormer.mlp_dropout = 0.0
    cfg.graphormer.input_dropout = 0.0
    cfg.graphormer.use_graph_token = True

    # TODO: move to posenc configs
    cfg.posenc_GraphormerBias = CN()
    cfg.posenc_GraphormerBias.enable = False
    cfg.posenc_GraphormerBias.node_degrees_only = False
    # I think dim_pe = 0 is only used for composed_encoders.py because here the PEs are not concatenated:
    cfg.posenc_GraphormerBias.dim_pe = 0
    cfg.posenc_GraphormerBias.num_spatial_types = None
    cfg.posenc_GraphormerBias.num_in_degrees = None
    cfg.posenc_GraphormerBias.num_out_degrees = None
    cfg.posenc_GraphormerBias.directed_graphs = False
    cfg.posenc_GraphormerBias.has_edge_attr = True
