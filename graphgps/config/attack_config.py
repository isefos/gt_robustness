from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('attack')
def dataset_cfg(cfg):
    """Attack config options.
    """
    # for transfer attacks from the robustness unit test
    cfg.robustness_unit_test = CN()
    cfg.robustness_unit_test.enable = False
    cfg.robustness_unit_test.load_best_model = True

    # for attack
    cfg.attack = CN()
    # whether to attack or not
    cfg.attack.enable = False
    # whether to also run a random baseline attack
    cfg.attack.run_random_baseline = True
    # whether to only run a random baseline attack
    cfg.attack.only_random_baseline = False
    # load the best validation model before attack or not
    cfg.attack.load_best_model = True
    # show progress bar or not
    cfg.attack.log_progress = True
    # whether to log all results (for each graph), or only the average in database observer
    # (can easily re-run with the saved perturbations to get per-graph results again)
    cfg.attack.only_return_avg = True
    # which split to attack, "train", "val", or "test"
    cfg.attack.split = "test"
    # set to 0 to attack all, n for only the first n
    cfg.attack.num_attacked_graphs = 0
    cfg.attack.epochs = 125
    cfg.attack.epochs_resampling = 100
    # how many gradient step updates before new edges are sampled
    cfg.attack.resample_period = 1
    cfg.attack.max_final_samples = 20
    cfg.attack.max_trials_sampling = 20
    cfg.attack.with_early_stopping = True
    cfg.attack.eps = 1e-7
    # Instead of initializing all new edges with eps, add some random variation
    cfg.attack.eps_init_noised = False
    # used for gradient clipping, to disable gradient clipping set to 0.0
    cfg.attack.max_edge_weight_update = 0.0
    cfg.attack.lr = 4_000
    cfg.attack.block_size = 2_000
    cfg.attack.e_budget = 0.1
    cfg.attack.minimum_budget = 0
    cfg.attack.is_undirected = True
    # what the predictions are for, node or graph
    cfg.attack.prediction_level = "graph"
    # 'train', 'masked', 'margin', 'prob_margin', or 'tanh_margin' (or callable)
    cfg.attack.loss = "train"
    # None (set to same as loss), or 'neg_accuracy'
    cfg.attack.metric = None
    # is important for node injection attacks, where graph is huge, but only some nodes get added, rest is disconnected
    cfg.attack.remove_isolated_components = False
    # None or int -> give an int (e.g. 0) to define that the root node of a graph will always be on the given index 
    # (used for removing isolated components)
    cfg.attack.root_node_idx = None
    # do we want to compute the node probability approximation or not (more important for node injection attacks)
    cfg.attack.node_prob_enable = False
    # how many iterations of the node probability approximation computation to do
    cfg.attack.node_prob_iterations = 3
    # compute the node probability approximation directly (faster) or in log space (better for numerical stability)
    cfg.attack.node_prob_log = True
    # will not attack a graph which is already incorrectly classified 
    # (faster, but if we want to transfer attack should keep False)
    cfg.attack.skip_incorrect_graph_classification = False
    # specifically for the CLUSTER dataset, to not sample edges to labeled nodes
    cfg.attack.cluster_sampling = False

    # For node injection attacks (sampling all edges independently):
    cfg.attack.node_injection = CN()
    # set True to do a node injection attack
    cfg.attack.node_injection.enable = False
    # when doing node injection attack, include nodes from train split to consider for injection
    cfg.attack.node_injection.from_train = True
    # when doing node injection attack, include nodes from val split to consider for injection
    cfg.attack.node_injection.from_val = True
    # when doing node injection attack, include nodes from test split to consider for injection
    cfg.attack.node_injection.from_test = True
    # whether the existing graph edges can be changed, or only new edges added
    cfg.attack.node_injection.allow_existing_graph_pert = True
    # sample only edges from existing nodes to new nodes, not from new to new
    cfg.attack.node_injection.sample_only_connected = False
    # when also sampling new-new edges (may be much more), can set a higher weight to sample edges from existing nodes (often minority, but more useful)
    cfg.attack.node_injection.existing_node_prob_multiplier = 1
    # for some dataset (e.g. UPFD) the root nodes are special, and each graph should only have one, therefore shouldn't be included for injection
    cfg.attack.node_injection.include_root_nodes = True
    # for tree datasets, when we inject a new node, we need to make sure it still has a tree structure
    cfg.attack.node_injection.sample_only_trees = False

    # node sampling, sample the nodes to inject first, then the edges to those nodes, more efficient in the sense that can sample more edges while adding less nodes
    cfg.attack.node_injection.node_sampling = CN()
    cfg.attack.node_injection.node_sampling.enable = False
    cfg.attack.node_injection.node_sampling.min_add_nodes = 100
    cfg.attack.node_injection.node_sampling.min_total_nodes = 1000
    cfg.attack.node_injection.node_sampling.max_block_size = 20_000

    # local attack
    cfg.attack.local = CN()
    cfg.attack.local.enable = False
    cfg.attack.local.num_victim_nodes = 5
    cfg.attack.local.sampling_direct_edge_weight = 5000
    cfg.attack.local.sampling_indirect_edge_weight = 30
    cfg.attack.local.sampling_other_edge_weight = 1
    cfg.attack.local.nettack = False

    # for transfer attack
    cfg.attack.transfer = CN()
    cfg.attack.transfer.enable = False
    cfg.attack.transfer.perturbation_path = ""

    # For specific models:

    cfg.attack.Graphormer = CN()
    # whether to use weighted degrees or not, if False, `combinations_degree` will be ignored
    cfg.attack.Graphormer.use_weighted_degrees = True
    # whether to calculate a simple sum of edge weights and do lin. interp. or use weight over all combinatorial degrees
    cfg.attack.Graphormer.combinations_degree = False
    # use reciprocal edge weight to find the shortest paths, if False, will ignore `sp_use_weighted` and `sp_use_gradient`
    cfg.attack.Graphormer.sp_find_weighted = True
    # how to invert the probabilities to make edge weights: inv or log or loglog
    cfg.attack.Graphormer.weight_function = "inv"
    # use the weighted distances for the found paths (if False will use the hop distance, and will ignore `sp_use_gradient`)
    cfg.attack.Graphormer.sp_use_weighted = True
    # use the gradient of the weighted shortest path distances
    cfg.attack.Graphormer.sp_use_gradient = True

    # TODO: should rename the Lap eigen-decomp. cfgs to "spectral" since they're also used by non-SAN models such as GPS...
    cfg.attack.SAN = CN()
    # weighted san (attackable), when edges partially true/fake, add to both true and fake attention mechanisms
    # (when False, all edges with p>0 will only be used in true attention mechanism without bias)
    cfg.attack.SAN.add_partially_fake_edges = True
    cfg.attack.SAN.partially_fake_edge_grad = True
    # enable or disable the backprop through eigendecomposition (either this or pert, not both)
    cfg.attack.SAN.enable_eig_backprop = False
    # how much space to put between repeated eigenvalues
    cfg.attack.SAN.eig_backprop_separation_pert = 1.0e-4
    # enable or disable the gradient using matrix perturbation approximation
    cfg.attack.SAN.enable_pert_grad = True
    # use as backwards pass differential approximation, i.e. use the actual values in the forward pass? 
    # (valid for both methods, pert and backprop)
    cfg.attack.SAN.pert_BPDA = False
    cfg.attack.SAN.eps_repeated_eigenvalue = 1.0e-5
    # probably good to use when BPDA is True
    cfg.attack.SAN.match_true_signs = False
    cfg.attack.SAN.match_true_eigenspaces = False
    # set the perturbation approximation of the first eigenvalue to zero (so that we don't try to optimize it)
    cfg.attack.SAN.set_first_pert_zero = False
    # Node Injection Attack with LapPE -> how to define the approximation perturbation
    cfg.attack.SAN.nia_pert = "full"  # full / half_weight / half_eps

    cfg.attack.GRIT = CN()
    # whether to compute a RRWP using the continuous edge probabilities or not, if False, will ignore grad_RRWP
    cfg.attack.GRIT.cont_RRWP = True
    # whether to compute a gradient through RRWP or not
    cfg.attack.GRIT.grad_RRWP = True
    # whether to weight the dummy edges by the cont. edge prob or not
    cfg.attack.GRIT.dummy_edge_weighting = True
    # whether to compute a continuous node degree, if False, will ignore grad_degree
    cfg.attack.GRIT.cont_degree = True
    # whether to compute a continuous node degree with gradient or not
    cfg.attack.GRIT.grad_degree = True

    cfg.attack.GPS = CN()
    cfg.attack.GPS.grad_MPNN = True

    # for "free" adversarial training -> all others are set by the cfg.attack configs
    cfg.train.adv = CN()
    # will be faster, as it only requires a single forward pass for batch, 
    # but some models do not support batched forward with adversarial input 
    # (because adv. input does not have pre-computed PEs, but could be fixed easily 
    # by implementing on the fly computation of PEs for a whole batch instead of only one graph) 
    cfg.train.adv.batched_train = True
    cfg.train.adv.e_budget = 0.15
    cfg.train.adv.block_size = 500
    cfg.train.adv.num_replays = 6
    cfg.train.adv.lr = 5000
    # for val
    cfg.train.adv.block_size_val = 1000
    cfg.train.adv.epochs_val = 16
    cfg.train.adv.epochs_resampling_val = 12
    cfg.train.adv.lr_val = 4000
    cfg.train.adv.max_final_samples = 4
    cfg.train.adv.max_num_graphs_val = 100
    # use the adversarial val eval for early stopping and selecting best model (or clean if False)
    cfg.train.adv.early_stopping_adv = True
    # whether to add the training dataset nodes in node injection validation eval
    cfg.train.adv.nia_include_train_in_val = True
