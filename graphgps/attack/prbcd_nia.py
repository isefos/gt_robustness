from typing import Callable, Optional, Tuple
from scipy.sparse.csgraph import minimum_spanning_tree
import torch
from torch import Tensor
from torch_geometric.utils import (
    to_undirected,
    to_scipy_sparse_matrix,
    coalesce,
)
from torch_geometric.data import Data
from graphgps.attack.prbcd import PRBCDAttack
from graphgps.attack.sampling import WeightedIndexSampler, get_connected_sampling_fun
from graphgps.transform.lap_eig import get_lap_decomp_stats
from torch_geometric.graphgym.config import cfg


# (predictions, labels, ids/mask) -> Tensor with one element
LOSS_TYPE = Callable[[Tensor, Tensor, Optional[Tensor]], Tensor]


class PRBCDAttackNI(PRBCDAttack):
    r"""Node injection attack via PRBCD
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__(model)

        # settings different sampling startegies
        self.existing_node_prob_multiplier = cfg.attack.node_injection.existing_node_prob_multiplier
        self.included_weighted = not self.existing_node_prob_multiplier == 1
        assert not (self.included_weighted and cfg.attack.node_injection.sample_only_connected), (
            "Either sample only from connected edges, or weighted (with relative weight for connected edges)"
        )
        self.sample_only_connected = cfg.attack.node_injection.sample_only_connected
        assert (
            cfg.attack.node_injection.allow_existing_graph_pert
            or self.sample_only_connected
            or self.included_weighted
        ), (
            "not allowing existing graph perturbations is only supported for "
            "weighted sampling (using existing_node_prob_multiplier != 1) or "
            "sampling only 'connected' edges (using sample_only_connected=True)"
        )
        self.allow_existing_graph_pert = cfg.attack.node_injection.allow_existing_graph_pert

        # will make sure that perturbations remain trees
        assert not (cfg.attack.node_injection.sample_only_trees and not self.is_undirected), (
            "Sampling only trees is only supported for undirected graphs"
        )
        self.sample_only_trees = cfg.attack.node_injection.sample_only_trees

    def _setup_sampling(self, *args, **kwargs):
        num_possible_edges = self._num_possible_edges(self.num_nodes, self.is_undirected)

        # 'Weighted' sampling
        if self.included_weighted:
            to_lin_idx = self._triu_to_linear_idx if self.is_undirected else self._full_to_linear_idx
            idx_to_included = None
            idx_between_included = None
            if self.included_weighted:
                # give edges to included nodes a higher probability of getting sampled
                edges_to_included = self._get_new_1hop_edges(
                    self.connected_nodes, self.num_nodes, self.is_undirected,
                )
                idx_to_included = to_lin_idx(self.num_nodes, edges_to_included)
            if not self.allow_existing_graph_pert:
                # don't allow the edges between to included nodes to change, only adding new edges
                edges_between_included = self._get_fully_connected_edges(
                    self.connected_nodes, self.is_undirected,
                )
                idx_between_included = to_lin_idx(self.num_nodes, edges_between_included)
            # Now use these for sampling
            self._weighted_sampler = WeightedIndexSampler(
                weighted_idx=idx_to_included,
                zero_idx=idx_between_included,
                weight=self.existing_node_prob_multiplier,
                max_index=num_possible_edges-1,
                output_device=self.device,
            )
            self.sample_edge_indices = lambda n: self._weighted_sampler.sample(n)
        
        # 'Only connected' sampling
        elif self.sample_only_connected:
            # assume the existing nodes are the first ones:
            num_new_nodes = self.num_nodes - self.num_connected_nodes
            num_existing_edges = self._num_possible_edges(self.num_connected_nodes, self.is_undirected)
            self.sample_edge_indices = get_connected_sampling_fun(
                allow_existing_graph_pert=self.allow_existing_graph_pert,
                is_undirected=self.is_undirected,
                n_ex_edges=num_existing_edges,
                n_ex_nodes=self.num_connected_nodes,
                n_new_nodes=num_new_nodes,
                device=self.device,
            )
        
        # 'Normal' sampling (all possible edges)
        else:
            self.sample_edge_indices = lambda n: torch.randint(num_possible_edges, (n, ), device=self.device)

    def _attack_self_setup(self, x, edge_index, random_baseline=False):
        super()._attack_self_setup(x, edge_index, random_baseline)
        self.num_injection_nodes = self.num_nodes - self.num_connected_nodes
        assert self.num_injection_nodes > 0, "use normal structure attack if no injection nodes are given"

    def _get_forward_data(self, x: Tensor, edge_index: Tensor, edge_weight: None | Tensor, discrete: bool) -> Data:
        # when node injection and laplacian eigen PE, get an off-perturbation
        edge_weight_off = None
        if not discrete and cfg.posenc_WLapPE.enable and cfg.attack.SAN.enable_pert_grad:
            node_inj_pert = cfg.attack.SAN.nia_pert
            if node_inj_pert != "full":

                if node_inj_pert == "half_weight":
                    block_weights_off = self.block_edge_weight.detach() / 2
                elif node_inj_pert == "half_eps":
                    block_weights_off = self.block_edge_weight.detach() - (self.coeffs['eps'] / 2)
                else:
                    raise ValueError(f"cfg.attack.SAN.nia_pert = {node_inj_pert} is not valid")

                _, edge_weight_off = self._get_modified_adj(
                    self.edge_index,
                    self.edge_weight,
                    self.block_edge_index,
                    block_weights_off,
                )
        # create a data, clone x, since it gets modified inplace in the forward pass
        data = Data(x=x.clone(), edge_index=edge_index, edge_attr=edge_weight, edge_weight_off=edge_weight_off)
        return data
    
    def _add_laplacian_info(self, data):
        num_nodes = data.x.size(0)
        num_nodes_added = num_nodes - self.num_connected_nodes
        if num_nodes_added == 0:
            data.lap_clean_edge_index = self.lap_edge_index
            data.lap_clean_edge_attr = self.lap_edge_attr
            data.E_clean = self.E_lap
            data.U_clean = self.U_lap.clone()
        else:
            assert num_nodes_added > 0, "Shouldn't be possible to have less during non-discrete"

            if cfg.attack.SAN.nia_pert == "full":

                device = self.lap_edge_index.device
                assert cfg.posenc_WLapPE.eigen.eigvec_norm  # not None, then eigenvalue of isolated is 1
                added_index = (torch.arange(num_nodes_added, device=device) + self.num_connected_nodes).repeat(2, 1)
                added_attr = self.lap_edge_attr.new_ones(num_nodes_added)
                data.lap_clean_edge_index, data.lap_clean_edge_attr = coalesce(
                    torch.cat([self.lap_edge_index, added_index], 1),
                    torch.cat([self.lap_edge_attr, added_attr], 0),
                )
                E_clean = torch.cat([self.E_lap, added_attr], 0) 
                U_clean = torch.block_diag(self.U_lap.clone(), torch.eye(num_nodes_added, device=device))
                idx = E_clean.argsort()
                E_clean = E_clean[idx]
                U_clean = U_clean[:, idx]
                data.E_clean = E_clean
                data.U_clean = U_clean

            else:
                # compute the laplcaian eigendecomposition of a slightly off-perturbed adjacency
                data.E_clean, data.U_clean, data.lap_clean_edge_index, data.lap_clean_edge_attr = get_lap_decomp_stats(
                    data.edge_index,
                    data.edge_weight_off,
                    data.x.size(0),
                    cfg.posenc_WLapPE.eigen.laplacian_norm,
                    max_freqs=cfg.posenc_WLapPE.eigen.max_freqs,
                    eigvec_norm=cfg.posenc_WLapPE.eigen.eigvec_norm,
                    pad_too_small=False,
                    need_full=True,
                    return_lap=True,
                )
        del data.edge_weight_off

    def _get_discrete_sampled_graph(self, sampled_block_edge_weight: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        returns:
        - edge_index and edge_weight -> define the entire discrete graph
        - discrete_block_edge_weight -> are the discrete values for just the block edges
        """
        edge_index, edge_weight = self._get_modified_adj(
            self.edge_index,
            self.edge_weight,
            self.block_edge_index,
            sampled_block_edge_weight,
        )
        assert torch.all(edge_weight > 0)
        
        num_removed_edges = 0
        if self.sample_only_trees:
            # ensure that we are sampling a tree
            g = to_scipy_sparse_matrix(edge_index, edge_attr=(1 / edge_weight.detach()))
            t = minimum_spanning_tree(g, overwrite=True).tocoo()
            edges_in_tree = set([frozenset([i, j]) for i, j in zip(*t.nonzero())])
            row = torch.from_numpy(t.row).to(dtype=torch.long, device=self.device)
            col = torch.from_numpy(t.col).to(dtype=torch.long, device=self.device)
            edge_index = torch.stack([row, col], dim=0)
            # discrete edges
            edge_weight = torch.ones((edge_index.size(1),), device=self.device)
            edge_index, edge_weight = to_undirected(edge_index, edge_weight)
            # find out which edges were removed -> remove from sampled_edge_weight
            # for edge in block_edge_index[sampled_edge_weight > 0] check if edge in edge_index
            # if not, it was removed to form the tree -> set the value in sampled_edge_weight to 0
            for i in sampled_block_edge_weight.nonzero().flatten().tolist():
                edge = frozenset(self.block_edge_index[:, i].flatten().tolist())
                if edge not in edges_in_tree:
                    sampled_block_edge_weight[i] = 0
                    num_removed_edges += 1

        discrete_block_edge_weight = (sampled_block_edge_weight > 0).float()
        return edge_index, discrete_block_edge_weight, num_removed_edges

    @staticmethod
    def _get_new_1hop_edges(
        sorted_included_node_idx: torch.Tensor,
        num_nodes: int,
        is_undirected: bool,
    ) -> torch.Tensor:
        """Returns edge_index including all edges between included nodes and injection nodes
        indices given as [i_row, j_col]
        when is_undirected: only one direction given with i_row < j_col
        """
        n_inc = sorted_included_node_idx.size(0)
        device = sorted_included_node_idx.device
        mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        mask[sorted_included_node_idx] = 0
        injection_nodes = torch.arange(0, num_nodes, dtype=torch.long, device=device)[mask]
        n_inj = injection_nodes.size(0)
        edges = torch.cat(
            (
                sorted_included_node_idx.repeat_interleave(n_inj)[None, :],
                injection_nodes.repeat(n_inc)[None, :],
            ),
            dim=0,
        )
        edges, _ = edges.sort(dim=0)
        if not is_undirected:
            edges = torch.cat((edges, edges[[1, 0], :]), dim=0)
        return edges

    @staticmethod
    def _get_fully_connected_edges(
        sorted_included_node_idx: torch.Tensor,
        is_undirected: bool,
    ) -> torch.Tensor:
        """Returns edge_index for all edges between included nodes
        indices given as [i_row, j_col]
        no self loops
        when is_undirected: only one direction given with i_row > j_col
        """
        n = sorted_included_node_idx.size(0)
        device = sorted_included_node_idx.device
        repeats_upper = torch.arange(n-1, -1, -1, dtype=torch.long, device=device)
        repeated_lower = torch.cat([sorted_included_node_idx[i:] for i in range(1, n)], dim=0)
        edges = torch.cat(
            (
                sorted_included_node_idx.repeat_interleave(repeats_upper)[None, :],
                repeated_lower[None, :]
            ),
            dim=0,
        )
        if not is_undirected:
            edges = torch.cat((edges, edges[[1, 0], :]), dim=0)
        return edges


class PRBCDAttackNISampling(PRBCDAttack):
    def __init__(self, model: torch.nn.Module):
        super(PRBCDAttackNI, self).__init__(model)
        self.sample_only_trees = cfg.attack.node_injection.sample_only_trees
        # will make sure that perturbations remain trees
        assert not (self.sample_only_trees and not self.is_undirected), (
            "Sampling only trees is only supported for undirected graphs"
        )
        
    def _setup_sampling(self, **kwargs):
        # TODO: sample the nodes first, then the edges to those nodes
        # self.samlple() will access the self.sampled_nodes to create a mapping of indexes
        # or just sample in 2D and transform to linear
        return
