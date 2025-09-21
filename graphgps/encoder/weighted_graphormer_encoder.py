import torch
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
from torch_geometric.utils import (
    to_dense_adj,
    to_scipy_sparse_matrix,
    scatter,
    remove_self_loops,
    subgraph,
)
from itertools import combinations
from graphgps.encoder.graphormer_encoder import add_graph_token, get_shortest_paths, get_discrete_degrees
from scipy.sparse import csgraph
import math
import functools


# Permutes from (batch, node, node, head) to (batch, head, node, node)
BATCH_HEAD_NODE_NODE = (0, 3, 1, 2)

# Inserts a leading 0 row and a leading 0 column with F.pad
INSERT_GRAPH_TOKEN = (1, 0, 1, 0)


def weighted_graphormer_pre_processing(data, attack_mode):
    """
    """
    distance: int = cfg.posenc_GraphormerBias.num_spatial_types
    directed_graphs: bool = cfg.posenc_GraphormerBias.directed_graphs
    num_in_degrees: int = cfg.posenc_GraphormerBias.num_in_degrees
    num_out_degrees: int = cfg.posenc_GraphormerBias.num_out_degrees
    node_degrees_only: bool = cfg.posenc_GraphormerBias.node_degrees_only            

    n = data.x.size(0)
    device = data.x.device

    if data.edge_attr is None:
        data.edge_attr = torch.ones((data.edge_index.size(1), ), device=device)

    if not attack_mode or not cfg.attack.Graphormer.use_weighted_degrees:
        in_degrees, out_degrees = get_discrete_degrees(data.edge_index, n)
    
        if out_degrees is None:
            data.degrees = in_degrees
        else:
            data.in_degrees = in_degrees
            data.out_degrees = out_degrees

    elif cfg.attack.Graphormer.combinations_degree:
        #in_degrees_max = torch.scatter(
        #    input=torch.zeros(n),
        #    value=1,
        #    index=data.edge_index[1, :],
        #    dim=0,
        #    reduce="add"
        #)
        #certain_edges = data.edge_index[:, data.edge_attr == 1]
        #in_degrees_min = torch.scatter(
        #    input=torch.zeros(n),
        #    value=1,
        #    index=certain_edges[1, :],
        #    dim=0,
        #    reduce="add"
        #)
        #ns = in_degrees_max - in_degrees_min
        #n_max = ns.max()
        # TODO: handle too large degree -> clamping to max
        #data.degree_indices, data.degree_weights = node_degree_weights_undirected(
        #    data.x.size(0), data.edge_index, data.edge_attr,
        #)
        raise NotImplementedError
    else:
        add_weighted_degrees_linear_interpolation_(data, n, device, num_in_degrees, num_out_degrees, directed_graphs)

    if node_degrees_only:
        return data
    
    data.graph_index = torch.cat(
        (
            torch.arange(n, dtype=torch.long, device=device).repeat_interleave(n)[None, :],
            torch.arange(n, dtype=torch.long, device=device).repeat(n)[None, :],
        ),
        dim=0,
    )

    max_distance = distance - 1
    if not (attack_mode and cfg.attack.Graphormer.sp_find_weighted):
        # just ignore the edge weights etirely
        spatial_types = get_shortest_paths(
            edge_index=data.edge_index,
            num_nodes=n,
            directed=directed_graphs,
            max_distance=max_distance,
        )
    elif cfg.attack.Graphormer.sp_use_weighted:
        # use the reciprocal of the edge weights to find the shortest paths
        # and use weighted path distance and do linear interpolation of the discrete distances
        spatial_types, spatial_types_weights = weighted_shortest_paths(
            use_weighted_gradient=cfg.attack.Graphormer.sp_use_gradient,
            num_nodes=n,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            max_distance=max_distance,
            directed_graphs=directed_graphs,
        )
        data.spatial_types_weights = spatial_types_weights
    else:
        # only use the edge weights to find the shortest paths,
        # but then use the actual hops of those paths as distance
        inv_edge_attr = probs_to_weights(data.edge_attr.detach())
        adj_weighted = to_scipy_sparse_matrix(data.edge_index, edge_attr=inv_edge_attr, num_nodes=n).tocsr()
        _, predecessors_np = csgraph.dijkstra(
            adj_weighted,
            directed=directed_graphs,
            return_predecessors=True,
            unweighted=False,
            limit=max_distance,
        )
        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=n).tocsr()
        distances_hop_np = csgraph.construct_dist_matrix(adj, predecessors_np, directed=directed_graphs)
        distances_hop_np[distances_hop_np > max_distance] = max_distance
        spatial_types = torch.tensor(distances_hop_np.reshape(n ** 2), dtype=torch.long, device=device)

    data.spatial_types = spatial_types
    return data


class WeightedBiasEncoder(torch.nn.Module):
    def __init__(self):
        """Implementation of the bias encoder of Graphormer modified for attackable model"""
        super().__init__()

        num_heads = cfg.graphormer.num_heads
        num_spatial_types = cfg.posenc_GraphormerBias.num_spatial_types

        self.spatial_encoder = torch.nn.Embedding(num_spatial_types, num_heads)
        if cfg.graphormer.use_graph_token:
            self.graph_token = torch.nn.Parameter(torch.zeros(1, num_heads, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.spatial_encoder.weight.data.normal_(std=0.02)
        if cfg.graphormer.use_graph_token:
            self.graph_token.data.normal_(std=0.02)

    def forward(self, data):
        """Computes the bias matrix that can be induced into multi-head attention
        via the attention mask.

        Adds the tensor ``attn_bias`` to the data object, optionally accounting
        for the graph token.
        """
        # To convert 2D matrices to dense-batch mode, one needs to decompose
        # them into index and value. One example is the adjacency matrix
        # but this generalizes actually to any 2D matrix
        if cfg.attack.Graphormer.sp_use_weighted and hasattr(data, "spatial_types_weights"):
            spatial_types: torch.Tensor = (
                data.spatial_types_weights[:, :, None] * self.spatial_encoder(data.spatial_types)
            ).sum(1)
        else:
            spatial_types: torch.Tensor = self.spatial_encoder(data.spatial_types)

        spatial_encodings = to_dense_adj(data.graph_index, data.batch, spatial_types)
        bias = spatial_encodings.permute(BATCH_HEAD_NODE_NODE)

        # during attack: adds bias for node's probability of being in graph
        if hasattr(data, "node_logprob"):
            n = data.node_logprob.size(0)
            assert n == bias.size(2) == bias.size(3)
            bias += data.node_logprob[None, None, None, :]

        elif hasattr(data, "node_prob"):
            n = data.node_prob.size(0)
            assert n == bias.size(2) == bias.size(3)
            # if p is so small that log(p) = -inf, gradient is undefined, so just set -inf for very small p
            l = torch.zeros_like(data.node_prob) - torch.inf
            min_prob_mask = data.node_prob > 1e-30
            l[min_prob_mask] = data.node_prob[min_prob_mask].log()
            bias += l[None, None, None, :]

        if cfg.graphormer.use_graph_token:
            bias = F.pad(bias, INSERT_GRAPH_TOKEN)
            bias[:, :, 1:, 0] = self.graph_token
            bias[:, :, 0, :] = self.graph_token
        
        B, H, N, _ = bias.shape
        data.attn_bias = bias.reshape(B * H, N, N)

        return data


class WeightedNodeEncoder(torch.nn.Module):
    def __init__(self, embed_dim):
        """Implementation of the node encoder of Graphormer modified for attackable model"""
        super().__init__()
        num_in_degree = cfg.posenc_GraphormerBias.num_in_degrees
        num_out_degree = cfg.posenc_GraphormerBias.num_out_degrees
        if cfg.posenc_GraphormerBias.directed_graphs:
            self.in_degree_encoder = torch.nn.Embedding(num_in_degree, embed_dim)
            self.out_degree_encoder = torch.nn.Embedding(num_out_degree, embed_dim)
        else:
            max_degree = max(num_in_degree, num_out_degree)
            self.degree_encoder = torch.nn.Embedding(max_degree, embed_dim)
        if cfg.graphormer.use_graph_token:
            self.graph_token = torch.nn.Parameter(torch.zeros(1, embed_dim))
        self.input_dropout = torch.nn.Dropout(cfg.graphormer.input_dropout)
        self.reset_parameters()

    def forward(self, data):
        if cfg.posenc_GraphormerBias.directed_graphs:
            if hasattr(data, "in_degrees"):
                # precomputed
                in_degree_encoding = self.in_degree_encoder(data.in_degrees)
                out_degree_encoding = self.out_degree_encoder(data.out_degrees)
            else:
                # continuous relaxation during attack
                if cfg.attack.Graphormer.combinations_degree:
                    # TODO: this doesn't quite seem right...
                    in_degree_encoding = data.degree_weights @ self.in_degree_encoder(data.degree_indices)
                    out_degree_encoding = data.degree_weights @ self.out_degree_encoder(data.degree_indices)
                else:
                    in_degree_encoding = (
                        data.in_degree_weights[:, :, None] * self.in_degree_encoder(data.in_degree_indices)
                    ).sum(1)
                    out_degree_encoding = (
                        data.out_degree_weights[:, :, None] * self.out_degree_encoder(data.out_degree_indices)
                    ).sum(1)
            degree_encoding = in_degree_encoding + out_degree_encoding
        else:
            if hasattr(data, "degrees"):
                # precomputed
                degree_encoding = self.degree_encoder(data.degrees)
            else:
                # continuous relaxation during attack
                if cfg.attack.Graphormer.combinations_degree:
                    degree_encoding = data.degree_weights @ self.degree_encoder(data.degree_indices)
                else:
                    degree_encoding = (
                        data.degree_weights[:, :, None] * self.degree_encoder(data.degree_indices)
                    ).sum(1)

        if data.x.size(1) > 0:
            data.x = data.x + degree_encoding
        else:
            data.x = degree_encoding

        if cfg.graphormer.use_graph_token:
            data = add_graph_token(data, self.graph_token)
        data.x = self.input_dropout(data.x)
        return data

    def reset_parameters(self):
        if cfg.posenc_GraphormerBias.directed_graphs:
            self.in_degree_encoder.weight.data.normal_(std=0.02)
            self.out_degree_encoder.weight.data.normal_(std=0.02)
        else:
            self.degree_encoder.weight.data.normal_(std=0.02)
        if cfg.graphormer.use_graph_token:
            self.graph_token.data.normal_(std=0.02)


class WeightedPreprocessing(torch.nn.Module):
    def forward(self, data):
        attack_mode = data.get("attack_mode", False)
        if attack_mode or data.get("spatial_types") is None:
            assert data.num_graphs == 1, "On the fly preprocessing only works for single graphs"
            data = weighted_graphormer_pre_processing(data, attack_mode)
        return data



@register_node_encoder("WeightedGraphormerBias")
class WeightedGraphormerEncoder(torch.nn.Sequential):
    def __init__(self, dim_emb, *args, **kwargs):
        assert not cfg.posenc_GraphormerBias.has_edge_attr, "Weighted graphormer cannot currently use edge attributes"
        encoders = [
            WeightedPreprocessing(),
            WeightedBiasEncoder(),
            WeightedNodeEncoder(dim_emb),
        ]
        if cfg.posenc_GraphormerBias.node_degrees_only:  # No attn. bias encoder
            encoders = encoders[1:]
        super().__init__(*encoders)


# utils for preprocessing:
        
# degrees:


def add_weighted_degrees_linear_interpolation_(data, n, device, num_in_degrees, num_out_degrees, directed_graphs):
    # TODO: refactor!
    in_degrees_weighted = torch.scatter_add(
        input=torch.zeros(n, device=device),
        src=data.edge_attr,
        index=data.edge_index[1, :],
        dim=0,
    )
    # clamp to maximum degree
    max_degree = max(num_in_degrees, num_out_degrees) - 1
    in_degrees_weighted[in_degrees_weighted > max_degree] = max_degree
    in_degrees_low = torch.floor(in_degrees_weighted).to(dtype=torch.long)
    in_degrees_high = torch.ceil(in_degrees_weighted).to(dtype=torch.long)
    weights_high = in_degrees_weighted - in_degrees_low
    weights_low = in_degrees_high - in_degrees_weighted
    weights_low[in_degrees_low == in_degrees_high] = 1
    # unweighted edges, remove zero degree encoding 
    # (never used during training, can't appear after discretization)
    weights_low[in_degrees_low == 0] = 0
    if directed_graphs:
        data.in_degree_indices = torch.cat((in_degrees_low[:, None], in_degrees_high[:, None]), dim=1)
        data.in_degree_weights = torch.cat((weights_low[:, None], weights_high[:, None]), dim=1)
        # now for out degrees as well
        out_degrees_weighted = torch.scatter_add(
            input=torch.zeros(n, device=device),
            src=data.edge_attr,
            index=data.edge_index[0, :],
            dim=0,
        )
        # clamp to maximum degree
        max_degree = max(num_in_degrees, num_out_degrees) - 1
        out_degrees_weighted[out_degrees_weighted > max_degree] = max_degree
        out_degrees_low = torch.floor(out_degrees_weighted).to(dtype=torch.long)
        out_degrees_high = torch.ceil(out_degrees_weighted).to(dtype=torch.long)
        weights_high = out_degrees_weighted - out_degrees_low
        weights_low = out_degrees_high - out_degrees_weighted
        weights_low[out_degrees_low == out_degrees_high] = 1
        # unweighted edges, remove zero degree encoding 
        # (never used during training, can't appear after discretization)
        weights_low[out_degrees_low == 0] = 0
        data.out_degree_indices = torch.cat((out_degrees_low[:, None], out_degrees_high[:, None]), dim=1)
        data.out_degree_weights = torch.cat((weights_low[:, None], weights_high[:, None]), dim=1)
    else:
        data.degree_indices = torch.cat((in_degrees_low[:, None], in_degrees_high[:, None]), dim=1)
        data.degree_weights = torch.cat((weights_low[:, None], weights_high[:, None]), dim=1)


def node_degree_weights_undirected(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    undirected_extra_checks: bool = False,
    remove_zero_degree: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    """
    # TODO: put on the correct device
    node_degree_info = get_node_degree_info(
        num_nodes=num_nodes,
        edge_index=edge_index,
        edge_weights=edge_weights,
        undirected=True,
        undirected_extra_checks=undirected_extra_checks,
    )
    node_stats, all_degrees = node_degree_info["undirected"]
    all_degree_indices, all_degree_weights = get_degree_weights(
        node_stats=node_stats, all_degrees=all_degrees, remove_zero_degree=remove_zero_degree,
    )
    return all_degree_indices, all_degree_weights


def get_node_degree_info(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    undirected: bool,
    undirected_extra_checks: bool = False,
) -> dict[str, tuple[list[tuple[int, int, torch.Tensor]], set[int]]]:
    """
    """
    if undirected:
        degree_edge_dims = {"undirected": 0}
        node_degree_info = {"undirected": ([], set())}
    else:
        degree_edge_dims = {"out": 0, "in": 1}
        node_degree_info = {"out": ([], set()), "in": ([], set())}
    for i in range(num_nodes):
        summed_degree = 0
        for degree_mode, idx in degree_edge_dims.items():
            edges_mask = edge_index[idx, :] == i
            max_degree = int(edges_mask.sum().item())
            summed_degree += max_degree
            uncertain_edge_mask = edges_mask * ((edges_mask * edge_weights) < 1)
            uncertain_edge_weights = edge_weights[uncertain_edge_mask]
            n_uncertain_edges = int(uncertain_edge_mask.sum().item())
            # assert max_in_degree - n_uncertain_edges == (edge_weights[edges_to_node_mask] == 1).sum().item()
            certain_degree = max_degree - n_uncertain_edges
            node_degree_info[degree_mode][0].append((certain_degree, n_uncertain_edges, uncertain_edge_weights))
            node_degree_info[degree_mode][1].update(range(certain_degree, max_degree+1))
            if undirected and undirected_extra_checks:
                opposite_edges_mask = edge_index[int(not idx), :] == i
                assert max_degree == opposite_edges_mask.sum(), "not undirected: different in / out degrees"
                assert torch.allclose(
                    edge_weights[edges_mask], edge_weights[opposite_edges_mask]
                ), "weight different for in/out"
        assert summed_degree > 0, "A totally isolated node should not be included"
    return node_degree_info


def get_degree_weights(
    node_stats: list[tuple[int, int, torch.Tensor]],
    all_degrees: set[int],
    remove_zero_degree: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    sorted_degrees = sorted(all_degrees)
    degree_index_map = {degree: i for i, degree in enumerate(sorted_degrees)}
    all_degree_indices = torch.tensor(sorted_degrees, dtype=torch.long)
    # dim = (N, (max_degree-min_degree)) -> maximally (N, N+1)
    all_degree_weights = torch.zeros((len(node_stats), all_degree_indices.size(0)))
    for node, (certain_degree, n, uncertain_edge_weights) in enumerate(node_stats):
        # can compute the indices as sum on top of min index safely, since all must be included
        min_idx = degree_index_map[certain_degree]
        if n == 0 and not certain_degree == 0:
            # a node with only certain edges
            all_degree_weights[node,  min_idx] += 1
            continue
        if certain_degree == 0 and n == 1:
            # a node with only a single uncertain edge
            # set degree to one, as we don't allow degree zero (does not appear in undirected graphs)
            all_degree_weights[node,  degree_index_map[1]] += uncertain_edge_weights.squeeze()
            continue
        weights_selection = torch.concat((1 - uncertain_edge_weights[None, :], uncertain_edge_weights[None, :]), dim=0)
        if n == 1:
            # only a single uncertain edge -> 2 options, can find directly:
            assert weights_selection.size(1) == 1
            all_degree_weights[node, [min_idx, min_idx+1]] += weights_selection.squeeze()
            continue
        # compute the combinations (n, k), start with the 0 and n: only a single combination
        all_degree_weights[node, min_idx] += torch.prod(weights_selection[0, :])
        all_degree_weights[node, min_idx+n] += torch.prod(weights_selection[1, :])
        if n > 2:
            # can skip for n == 2, uses symmetry for k and n-k
            # num_combinations = 1
            indices = tuple(range(n))
            max_k = (n + 1) // 2
            for k in range(1, max_k):
                # num_combinations *= (n + 1 - k) // k
                for comb in combinations(indices, k):
                    mask = torch.zeros_like(weights_selection).to(dtype=torch.bool)
                    mask[0, :] = 1
                    mask[0, comb] = 0
                    mask[1, comb] = 1
                    # for k
                    all_degree_weights[node, min_idx+k] += torch.prod((mask * weights_selection).sum(dim=0))
                    # for n-k
                    all_degree_weights[node, min_idx+n-k] += torch.prod((~mask * weights_selection).sum(dim=0))
        if n % 2 == 0:
            # even -> the last entry in the 'middle' is still missing
            # TODO: does it make a difference if we compute it in this simplified manner for the gradients?
            all_degree_weights[node, min_idx+n//2] += 1 - all_degree_weights[node, :].sum()
    if remove_zero_degree and sorted_degrees[0] == 0:
        # don't allow the option of no edges existing, since that would lead to an isolated node
        all_degree_indices = all_degree_indices[1:]
        all_degree_weights = all_degree_weights[:, 1:]
    return all_degree_indices, all_degree_weights


# shortest paths:


@functools.cache
def get_base(l: str) -> float:
    eps = cfg.attack.eps
    assert cfg.posenc_GraphormerBias.num_spatial_types >= 3
    dmax = cfg.posenc_GraphormerBias.num_spatial_types - 1.001
    if l == "log":
        return math.log(math.pow(eps, (1 / (1 - dmax))))
    elif l == "loglog":
        return math.log(math.pow((1 - math.log10(eps)), (1 / (dmax - 1))))
    raise ValueError


def probs_to_weights(probs: torch.Tensor) -> torch.Tensor:
    if cfg.attack.Graphormer.weight_function == "inv":
        weights = 1 / probs
    elif cfg.attack.Graphormer.weight_function == "log":
        weights = 1 - (probs.log() / get_base("log"))
    elif cfg.attack.Graphormer.weight_function == "loglog":
        weights = 1 + ((1 - probs.log10()).log() / get_base("loglog"))
    else:
        raise ValueError(
            f"Unsupported `cfg.attack.Graphormer.weight_function`: {cfg.attack.Graphormer.weight_function}"
        )
    return weights


def weighted_shortest_paths(
    use_weighted_gradient: bool,
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    max_distance: int,
    directed_graphs: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = edge_index.device
    # when using the weighted path distance, we can prune away all nodes that we know have a distance > max_distance
    num_distances = num_nodes ** 2
    max_edge_probs = get_node_max_edge_prob(
        edge_index=edge_index,
        edge_weights=edge_attr,
        num_nodes=num_nodes,
        is_undirected=not directed_graphs,
    )
    # find the minimum shortest path for each node
    min_shortest_path = probs_to_weights(max_edge_probs)
    prune_mask = min_shortest_path <= max_distance
    p_num_nodes = int(prune_mask.sum().item())
    distances_prune_mask = (prune_mask[:, None] * prune_mask[None, :]).reshape(num_distances)
    p_edge_index, p_edge_attr = subgraph(prune_mask, edge_index, edge_attr, relabel_nodes=True)

    distances_weighted = distances_shortest_weighted_paths(
        use_weighted_gradient=use_weighted_gradient,
        num_nodes=p_num_nodes,
        edge_index=p_edge_index,
        edge_attr=p_edge_attr,
        max_distance=max_distance,
        directed_graphs=directed_graphs,
    )

    debug = False
    if debug:
        d_p = torch.full((num_distances, ), max_distance, dtype=torch.float32, device=device)
        # set diagonal to zero
        _d = torch.arange(num_nodes, dtype=torch.long, device=device)
        diag_idx = num_nodes * _d + _d
        d_p[diag_idx] = 0
        d_p[distances_prune_mask] = distances_weighted
        d_not_p = distances_shortest_weighted_paths(
            use_weighted_gradient=use_weighted_gradient,
            num_nodes=num_nodes,
            edge_index=edge_index,
            edge_attr=edge_attr,
            max_distance=max_distance,
            directed_graphs=directed_graphs,
        )
        assert torch.allclose(d_p, d_not_p)

    # dim1: 0 for low, 1 for high
    spatial_types = torch.full((num_distances, 2), max_distance, dtype=torch.long, device=device)
    spatial_types_weights = torch.zeros((num_distances, 2), device=device)
    spatial_types_weights[:, 0] = 1

    # linear interpolation weights of the discrete distances inserted into the total tensors for all distances
    spatial_types[distances_prune_mask, 1] = torch.ceil(distances_weighted).to(dtype=torch.long)
    spatial_types[distances_prune_mask, 0] = torch.floor(distances_weighted).to(dtype=torch.long)
    spatial_types_weights[distances_prune_mask, 1] = distances_weighted - spatial_types[distances_prune_mask, 0]
    p_low = spatial_types[distances_prune_mask, 1] - distances_weighted
    p_low[spatial_types[distances_prune_mask, 0] == spatial_types[distances_prune_mask, 1]] = 1
    spatial_types_weights[distances_prune_mask, 0] = p_low

    assert torch.all(spatial_types_weights >= 0)

    return spatial_types, spatial_types_weights


def distances_shortest_weighted_paths(
    use_weighted_gradient: bool,
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    max_distance: int,
    directed_graphs: bool,
):
    num_distances = num_nodes ** 2
    inv_edge_attr = probs_to_weights(edge_attr)
    adj_weighted = to_scipy_sparse_matrix(edge_index, edge_attr=inv_edge_attr.detach(), num_nodes=num_nodes).tocsr()
    distances_weighted_np, predecessors_np = csgraph.dijkstra(
        adj_weighted,
        directed=directed_graphs,
        return_predecessors=True,
        unweighted=False,
        limit=max_distance,
    )
    distances_weighted_np = distances_weighted_np.reshape(num_distances)
    clamped_distance_mask_np = distances_weighted_np > max_distance
    distances_weighted_np[clamped_distance_mask_np] = max_distance
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()
    distances_hop_np = csgraph.construct_dist_matrix(adj, predecessors_np, directed=directed_graphs)
    distances_hop_np = distances_hop_np.reshape(num_distances)
    distances_hop_np[clamped_distance_mask_np] = max_distance
    max_hops = int(distances_hop_np.max())

    device = edge_index.device
    if use_weighted_gradient:
        clamped_distance_mask = torch.tensor(clamped_distance_mask_np, dtype=torch.bool, device=device)
        distances_weighted = reconstruct_weighted_distances_over_path(
            inv_edge_attr=inv_edge_attr,
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_distances=num_distances,
            max_distance=max_distance,
            max_hops=max_hops,
            clamped_distance_mask=clamped_distance_mask,
            predecessors_np=predecessors_np,
            distances_weighted_np=distances_weighted_np,
        )
    else:
        distances_weighted = torch.tensor(distances_weighted_np, dtype=torch.float32, device=device)

    return distances_weighted


def reconstruct_weighted_distances_over_path(
    inv_edge_attr: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    num_distances: int,
    max_distance: int,
    max_hops: int,
    clamped_distance_mask: torch.Tensor,
    predecessors_np,
    distances_weighted_np,
):
    device = edge_index.device
    # get dense inv adj for easier indexing in path reconstruction
    # TODO: instead, use SparseTensor and index with 2D indices
    inv_adj = scatter(
        inv_edge_attr,
        num_nodes * edge_index[0] + edge_index[1],
        dim=0,
        dim_size=num_distances,
        reduce='sum',
    )

    distances_weighted = torch.zeros((num_distances, ), device=device)
    distances_weighted[clamped_distance_mask] = max_distance
    predecessors = torch.tensor(predecessors_np.reshape((num_distances, )), dtype=torch.long, device=device)
    has_next_mask = predecessors >= 0

    # note that "current" and "next" are essentialy reversed here,
    # we start at the end of the path (index j -> row) and make our way to the start (index i -> col)
    cols = torch.arange(num_nodes, dtype=torch.long, device=device).repeat_interleave(num_nodes)[has_next_mask]
    current_nodes = torch.arange(num_nodes, dtype=torch.long, device=device).repeat(num_nodes)[has_next_mask]
    adj_pairs_idx = num_nodes * cols + current_nodes

    for _ in range(max_hops):
        next_nodes = predecessors[num_nodes * cols + current_nodes]
        distances_weighted[adj_pairs_idx] += inv_adj[num_nodes * next_nodes + current_nodes]
        current_nodes = next_nodes
        path_incomplete_mask = current_nodes != cols
        current_nodes = current_nodes[path_incomplete_mask]
        cols = cols[path_incomplete_mask]
        adj_pairs_idx = adj_pairs_idx[path_incomplete_mask]

    assert current_nodes.numel() == 0
    assert torch.allclose(distances_weighted, torch.tensor(distances_weighted_np, dtype=torch.float32, device=device))
    return distances_weighted


def get_node_max_edge_prob(
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    num_nodes: int,
    is_undirected: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    """
    edge_index_nsl, edge_weights_nsl = remove_self_loops(edge_index, edge_weights)

    if is_undirected:
        max_edge_weight = scatter(edge_weights_nsl, edge_index_nsl[0, :], dim=0, dim_size=num_nodes, reduce='max')
    else:
        max_edge_weight = scatter(
            torch.cat((edge_weights_nsl, edge_weights_nsl), dim=0),
            edge_index_nsl.flatten(),
            dim=0,
            dim_size=num_nodes,
            reduce='max',
        )

    return max_edge_weight