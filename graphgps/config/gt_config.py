from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_gt')
def set_cfg_gt(cfg):
    """Configuration for Graph Transformer-style models, e.g.:
    - Spectral Attention Network (SAN) Graph Transformer.
    - "vanilla" Transformer / Performer.
    - General Powerful Scalable (GPS) Model.
    """

    # Positional encodings argument group
    cfg.gt = CN()

    # Type of Graph Transformer layer to use
    cfg.gt.layer_type = 'SANLayer'

    # Number of Transformer layers in the model
    cfg.gt.layers = 3

    # Number of attention heads in the Graph Transformer
    cfg.gt.n_heads = 8

    # Size of the hidden node and edge representation
    cfg.gt.dim_hidden = 64

    # Full attention SAN transformer including all possible pairwise edges
    cfg.gt.full_graph = True

    # SAN real vs fake edge attention weighting coefficient
    cfg.gt.gamma = 1e-5

    # gamma parameter learnable or not
    cfg.gt.gamma_learnable = False

    # Histogram of in-degrees of nodes in the training set used by PNAConv.
    # Used when `gt.layer_type: PNAConv+...`. If empty it is precomputed during
    # the dataset loading process.
    cfg.gt.pna_degrees = []

    # Dropout in feed-forward module.
    cfg.gt.dropout = 0.0

    # Dropout in self-attention.
    cfg.gt.attn_dropout = 0.0

    cfg.gt.layer_norm = False

    cfg.gt.batch_norm = True

    # makes no difference for GRIT (always includes residual)
    # TODO: make all implementations more general to include option
    cfg.gt.residual = True

    # GRIT
    # go into GritTransformerLayer
    cfg.gt.bn_momentum = 0.1 # 0.01
    cfg.gt.bn_no_runner = False
    cfg.gt.update_e = True
    
    cfg.gt.attn = CN()

    # goes into rrwp linear edge encoder as pad_to_full_graph
    cfg.gt.attn.full_attn = True
    
    # go into GritTransformerLayer
    cfg.gt.attn.deg_scaler = True
    cfg.gt.attn.use_bias = False
    cfg.gt.attn.clamp = 5.
    cfg.gt.attn.act = "relu"
    cfg.gt.attn.norm_e = True
    cfg.gt.attn.O_e = True
    cfg.gt.attn.edge_enhance = True

    # Polynormer
    cfg.gt.polynormer = CN()
    # TODO: input dropout, remove from here once more general setting is implemented
    cfg.gt.polynormer.dropout_node_input = 0.0
    cfg.gt.polynormer.beta = -1.
    # TODO: move to gnn to be able to set BN xor LN
    # essentially always set to false in original polynormer code 
    cfg.gt.polynormer.local_pre_layer_norm = False
    # always set to true in original polynormer code
    cfg.gt.polynormer.qk_shared = True
