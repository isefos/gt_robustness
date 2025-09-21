from collections import defaultdict
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch_geometric.utils import (
    coalesce,
    to_undirected,
)
from torch_geometric.data import Data
from graphgps.attack.preprocessing import remove_isolated_components
from graphgps.attack.sampling import WeightedIndexSampler2
from graphgps.transform.lap_eig import get_lap_decomp_stats, get_repeated_eigenvalue_slices
from torch_geometric.graphgym.loss import compute_loss
import logging
from torch_geometric.graphgym.config import cfg


class PRBCDAttack(torch.nn.Module):
    r"""The Projected Randomized Block Coordinate Descent (PRBCD) adversarial
    attack from the `Robustness of Graph Neural Networks at Scale
    <https://www.cs.cit.tum.de/daml/robustness-of-gnns-at-scale>`_ paper.

    This attack uses an efficient gradient based approach that (during the
    attack) relaxes the discrete entries in the adjacency matrix
    :math:`\{0, 1\}` to :math:`[0, 1]` and solely perturbs the adjacency matrix
    (no feature perturbations). Thus, this attack supports all models that can
    handle weighted graphs that are differentiable w.r.t. these edge weights,
    *e.g.*, :class:`~torch_geometric.nn.conv.GCNConv` or
    :class:`~torch_geometric.nn.conv.GraphConv`. For non-differentiable models
    you might need modifications, e.g., see example for
    :class:`~torch_geometric.nn.conv.GATConv`.

    The memory overhead is driven by the additional edges (at most
    :attr:`block_size`). For scalability reasons, the block is drawn with
    replacement and then the index is made unique. Thus, the actual block size
    is typically slightly smaller than specified.

    This attack can be used for both global and local attacks as well as
    test-time attacks (evasion) and training-time attacks (poisoning). Please
    see the provided examples.

    This attack is designed with a focus on node- or graph-classification,
    however, to adapt to other tasks you most likely only need to provide an
    appropriate loss and model. However, we currently do not support batching
    out of the box (sampling needs to be adapted).

    .. note::
        For examples of using the PRBCD Attack, see
        `examples/contrib/rbcd_attack.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        contrib/rbcd_attack.py>`_
        for a test time attack (evasion) or
        `examples/contrib/rbcd_attack_poisoning.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        contrib/rbcd_attack_poisoning.py>`_
        for a training time (poisoning) attack.

    Args:
        model (torch.nn.Module): The GNN module to assess.
        is_undirected (bool, optional): If :obj:`True` the graph is
            assumed to be undirected. (default: :obj:`True`)
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()

        self.model = model
        self.block_size = cfg.attack.block_size
        self.epochs = cfg.attack.epochs

        loss = cfg.attack.loss
        if isinstance(loss, str):
            self.loss = self._get_loss(loss)
        else:
            raise ValueError(f'Unknown loss `{loss}`')

        self.is_undirected = cfg.attack.is_undirected
        self.log = cfg.attack.log_progress

        metric = cfg.attack.metric
        if metric is None:
            self.metric = self.loss
        elif isinstance(metric, str):
            if metric == 'neg_accuracy':
                self.metric = self._neg_accuracy_metric
            else:
                self.metric = self._get_loss(metric, "metric")
        else:
            raise ValueError(f'Unknown metric `{loss}`')

        self.epochs_resampling = cfg.attack.epochs_resampling
        self.resample_period = cfg.attack.resample_period
        self.lr = cfg.attack.lr

        self.coeffs = {
            'max_final_samples': cfg.attack.max_final_samples,
            'max_trials_sampling': cfg.attack.max_trials_sampling,
            'with_early_stopping': cfg.attack.with_early_stopping,
            'eps': cfg.attack.eps,
        }

        # gradient clipping
        self.max_edge_weight_update = cfg.attack.max_edge_weight_update

    def _get_loss(self, loss: str, name: str = "loss"):
        if loss == 'masked':
            return self._masked_cross_entropy
        elif loss == 'margin':
            return partial(self._margin_loss, reduce='mean')
        elif loss == 'prob_margin':
            return self._probability_margin_loss
        elif loss == 'tanh_margin':
            return self._tanh_margin_loss
        elif loss == 'train':
            return self._train_loss
        elif loss == 'raw_prediction':
            return self._raw_prediction_loss
        elif loss == 'reflected_cross_entropy':
            return self._reflected_cross_entropy_loss
        else:
            raise ValueError(f'Unknown {name} `{loss}`')

    def _setup_sampling(
        self,
        x: Tensor,
        victim_node_idx: None | int = None,
        **kwargs,
    ):
        assert self.num_nodes == self.num_connected_nodes, "structure attack should not include isolated nodes"
        num_possible_edges = self._num_possible_edges(self.num_nodes, self.is_undirected)
        if cfg.attack.local.enable:
            # for more efficient attack, sample the edges that are more likely to affect the local victim node
            assert self.is_undirected  # simplifies finding neighbors
            assert victim_node_idx is not None
            victim_node = torch.tensor([victim_node_idx], dtype=torch.long, device="cpu")
            other_nodes = torch.tensor(
                [i for i in range(self.num_nodes) if i != victim_node_idx],
                dtype=torch.long, device="cpu",
            )
            edges_to_victim_node = torch.cat(
                (
                    victim_node.repeat(other_nodes.size(0))[None, :],
                    other_nodes[None, :],
                ),
                dim=0,
            ).sort(0)[0]
            victim_edge_idx = self._triu_to_linear_idx(self.num_nodes, edges_to_victim_node).unique(sorted=True)
            if cfg.attack.local.sampling_indirect_edge_weight != cfg.attack.local.sampling_other_edge_weight:
                neighbor_nodes: torch.Tensor = self.edge_index[1, :][self.edge_index[0] == victim_node_idx]
                assert victim_node_idx not in neighbor_nodes, "Unexpected: Victim node has a self-loop with itself."
                edges_to_neighbor_nodes = torch.cat(
                    (
                        neighbor_nodes.repeat_interleave(other_nodes.size(0))[None, :],
                        other_nodes.repeat(neighbor_nodes.size(0))[None, :],
                    ),
                    dim=0,
                )
                # remove self loops
                edges_to_neighbor_nodes = edges_to_neighbor_nodes[:, edges_to_neighbor_nodes[0, :] != edges_to_neighbor_nodes[1, :]]
                # sort
                edges_to_neighbor_nodes = edges_to_neighbor_nodes.sort(0)[0]
                neighbor_edge_idx = self._triu_to_linear_idx(self.num_nodes, edges_to_neighbor_nodes).unique(sorted=True)
            else:
                neighbor_edge_idx = torch.tensor([], dtype=torch.long, device="cpu")
            # TODO: could still also include the CLUSTER constrained option of not allowing label nodes
            self._weighted_sampler = WeightedIndexSampler2(
                {
                    cfg.attack.local.sampling_direct_edge_weight: victim_edge_idx,
                    cfg.attack.local.sampling_indirect_edge_weight: neighbor_edge_idx,
                },
                default_weight=cfg.attack.local.sampling_other_edge_weight,
                max_index=num_possible_edges-1,
                output_device=self.device,
            )
            self.sample_edge_indices = lambda n: self._weighted_sampler.sample(n)
        # For global attacks
        elif cfg.attack.cluster_sampling:
            # for the CLUSTER dataset, we don't allow modifying edges to the labeled nodes
            assert self.is_undirected
            label_nodes = torch.nonzero(x[:, 1:].sum(1)).flatten().cpu()
            assert label_nodes.size(0) == 6
            all_nodes = torch.arange(self.num_nodes, dtype=torch.long, device="cpu")
            edges_to_label_nodes = torch.cat(
                (
                    label_nodes.repeat_interleave(self.num_nodes)[None, :],
                    all_nodes.repeat(6)[None, :],
                ),
                dim=0,
            )
            # remove the 6 self loops
            edges_to_label_nodes = edges_to_label_nodes[:, edges_to_label_nodes[0, :] != edges_to_label_nodes[1, :]]
            # make sure row is smaller than col
            edges_to_label_nodes = edges_to_label_nodes.sort(0)[0]
            # unique to eliminate the few duplicates between labeled nodes
            lin_idx = self._triu_to_linear_idx(self.num_nodes, edges_to_label_nodes).unique(sorted=True)
            num_possible_edges = self._num_possible_edges(self.num_nodes, self.is_undirected)
            self._weighted_sampler = WeightedIndexSampler2(
                {0: lin_idx},
                default_weight=1,
                max_index=num_possible_edges-1,
                output_device=self.device,
            )
            self.sample_edge_indices = lambda n: self._weighted_sampler.sample(n)
        else:
            # 'Normal' sampling (all possible edges)
            self.sample_edge_indices = lambda n: torch.randint(num_possible_edges, (n, ), device=self.device)

    def _setup_clean_lap_eigen(self, edge_index):
        self.E_lap, self.U_lap, self.lap_edge_index, self.lap_edge_attr = get_lap_decomp_stats(
            edge_index,
            None,
            self.num_connected_nodes,
            cfg.posenc_WLapPE.eigen.laplacian_norm,
            max_freqs=cfg.posenc_WLapPE.eigen.max_freqs,
            eigvec_norm=cfg.posenc_WLapPE.eigen.eigvec_norm,
            pad_too_small=False,
            need_full=True,
            return_lap=True,
        )

    def _attack_self_setup(self, x, edge_index, random_baseline=False):
        self.model.eval()
        self.device = x.device
        self.edge_index = edge_index.cpu().clone()
        self.edge_weight = torch.ones(edge_index.size(1), device="cpu")
        self.num_nodes = x.size(0)
        self.connected_nodes: torch.Tensor = self.edge_index.unique(sorted=True)
        self.num_connected_nodes = self.connected_nodes.size(0)
        assert self.connected_nodes[-1] == self.num_connected_nodes - 1, "some nodes of the clean graph are isolated"
        # get the clean laplacian eigendecomposition
        if cfg.posenc_WLapPE.enable and cfg.attack.SAN.enable_pert_grad and not random_baseline:
            assert self.is_undirected
            self._setup_clean_lap_eigen(edge_index)

    def attack(
        self,
        x: Tensor,
        edge_index: Tensor,
        labels: Tensor,
        budget: int,
        idx_attack: Optional[Tensor] = None,
        local_victim_node: None | int = None,
        tb_writer: None | SummaryWriter = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """Attack the predictions for the provided model and graph.

        A subset of predictions may be specified with :attr:`idx_attack`. The
        attack is allowed to flip (i.e. add or delete) :attr:`budget` edges and
        will return the strongest perturbation it can find. It returns both the
        resulting perturbed :attr:`edge_index` as well as the perturbations.

        Args:
            x (torch.Tensor): The node feature matrix.
            edge_index (torch.Tensor): The edge indices.
            labels (torch.Tensor): The labels.
            budget (int): The number of allowed perturbations (i.e.
                number of edges that are flipped at most).
            idx_attack (torch.Tensor, optional): Filter for predictions/labels.
                Shape and type must match that it can index :attr:`labels`
                and the model's predictions.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`)
        """
        assert kwargs.get('edge_weight') is None
        self._attack_self_setup(x, edge_index)

        num_possible_edges = self._num_possible_edges(self.num_nodes, self.is_undirected)
        prev_block_size = self.block_size
        if self.block_size > num_possible_edges:
            self.block_size = num_possible_edges

        self._setup_sampling(x=x, victim_node_idx=local_victim_node)
            
        # For collecting attack statistics
        self.attack_statistics = defaultdict(list)

        # Prepare attack and define `self.iterable` to iterate over
        step_sequence = self._prepare(budget)

        # Loop over the epochs (Algorithm 1, line 5)
        for step in tqdm(step_sequence, disable=not self.log, desc='Attack'):
            loss, gradient = self._forward_and_gradient(x, labels, idx_attack, **kwargs)

            scalars = self._update(step, gradient, x, labels, budget, loss, idx_attack, tb_writer, **kwargs)
            self._append_statistics(scalars)

        perturbed_edge_index, flipped_edges = self._close(x, labels, budget, idx_attack, **kwargs)

        assert flipped_edges.size(1) <= budget, (
            f'# perturbed edges {flipped_edges.size(1)} exceeds budget {budget}'
        )

        self.block_size = prev_block_size

        return perturbed_edge_index, flipped_edges
    
    def attack_random_baseline(
        self,
        x: Tensor,
        edge_index: Tensor,
        labels: Tensor,
        budget: int,
        idx_attack: Optional[Tensor] = None,
        local_victim_node: None | int = None,
        tb_writer: None | SummaryWriter = None,  # not used
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """Attack the predictions for the provided model and graph.

        A subset of predictions may be specified with :attr:`idx_attack`. The
        attack is allowed to flip (i.e. add or delete) :attr:`budget` edges and
        will return the strongest perturbation it can find. It returns both the
        resulting perturbed :attr:`edge_index` as well as the perturbations.

        Args:
            x (torch.Tensor): The node feature matrix.
            edge_index (torch.Tensor): The edge indices.
            labels (torch.Tensor): The labels.
            budget (int): The number of allowed perturbations (i.e.
                number of edges that are flipped at most).
            idx_attack (torch.Tensor, optional): Filter for predictions/labels.
                Shape and type must match that it can index :attr:`labels`
                and the model's predictions.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`)
        """
        assert kwargs.get('edge_weight') is None
        self._attack_self_setup(x, edge_index, random_baseline=True)
        self._setup_sampling(x=x, victim_node_idx=local_victim_node)
        best_metric = float('-Inf')
        best_edge_index = None

        for _ in tqdm(
            range(self.epochs + self.coeffs['max_final_samples']), disable=not self.log, desc='Random attack',
        ):
            self.current_block = self.sample_edge_indices(budget)
            self.current_block = torch.unique(self.current_block, sorted=True)

            if self.is_undirected:
                self.block_edge_index = self._linear_to_triu_idx(self.num_nodes, self.current_block)
            else:
                self.block_edge_index = self._linear_to_full_idx(self.num_nodes, self.current_block)
                self._filter_self_loops_in_block(with_weight=False)

            self.block_edge_weight = torch.full(self.current_block.shape, 1, device=self.device)
            self.block_edge_weight.requires_grad = False

            edge_index, _ = self._get_modified_adj(
                self.edge_index,
                self.edge_weight,
                self.block_edge_index,
                self.block_edge_weight,
            )

            prediction = self._forward(x, edge_index, None, discrete=True, **kwargs)
            metric = self.metric(prediction, labels, idx_attack)

            if metric > best_metric:
                best_edge_index = edge_index.cpu().clone()
                best_block_edge_index = self.block_edge_index.cpu().clone()
                best_metric = metric
        
        return best_edge_index.to(self.device), best_block_edge_index.to(self.device)

    def _prepare(self, budget: int) -> Iterable[int]:
        """Prepare attack."""
        if self.block_size <= budget:
            raise ValueError(
                f'The search space size ({self.block_size}) must be greater than the number of permutations ({budget})'
            )
        # For early stopping (not explicitly covered by pseudo code)
        self.best_metric = float('-Inf')
        self.best_loss = float('-Inf')
        # Sample initial search space (Algorithm 1, line 3-4)
        self._sample_random_block(budget)
        steps = range(self.epochs)
        return steps

    @torch.no_grad()
    def _update(
        self,
        epoch: int,
        gradient: Tensor,
        x: Tensor,
        labels: Tensor,
        budget: int,
        loss: Tensor,
        idx_attack: Optional[Tensor] = None,
        tb_writer: None | SummaryWriter = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Update edge weights given gradient."""
        # Gradient update step (Algorithm 1, line 7)
        lr = self._get_learning_rate(budget, epoch)
        self.block_edge_weight = self._update_edge_weights(self.block_edge_weight, gradient, lr)

        # For monitoring
        pmass_update = torch.clamp(self.block_edge_weight, 0, 1)
        # Projection to stay within relaxed `L_0` budget
        # (Algorithm 1, line 8)
        self.block_edge_weight = self._project(budget, self.block_edge_weight, self.coeffs['eps'])

        # For monitoring
        scalars = dict(
            prob_mass_after_update=pmass_update.sum().item(),
            prob_mass_after_update_max=pmass_update.max().item(),
            prob_mass_after_projection=self.block_edge_weight.sum().item(),
            prob_mass_after_projection_nonzero_weights=(self.block_edge_weight > self.coeffs['eps']).sum().item(),
            prob_mass_after_projection_max=self.block_edge_weight.max().item(),
        )
        scalars['lr'] = lr
        scalars['max_abs_grad'] = gradient.abs().max().item()
        scalars['loss'] = loss.item()

        if not self.coeffs['with_early_stopping']:
            # Resampling of search space, even when not saving the best
            if epoch < self.epochs_resampling - 1 and (epoch + 1) % self.resample_period == 0:
                self._resample_random_block(budget)
            self._write_scalars_to_tb(tb_writer, scalars, epoch)            
            return scalars

        # Calculate metric after the current epoch (overhead
        # for monitoring and early stopping)
        topk_block_edge_weight = torch.zeros_like(self.block_edge_weight)
        topk_indices = torch.topk(self.block_edge_weight, budget).indices
        topk_block_edge_weight[topk_indices] = self.block_edge_weight[topk_indices]
        
        edge_index = self._get_discrete_sampled_graph(topk_block_edge_weight)[0]
        
        prediction = self._forward(x, edge_index, None, discrete=True, **kwargs)
        loss_discrete = self.loss(prediction, labels, idx_attack)
        metric = self.metric(prediction, labels, idx_attack)

        # Save best epoch for early stopping
        # (not explicitly covered by pseudo code)
        if metric >= self.best_metric and loss_discrete > self.best_loss:
            self.best_metric = metric
            self.best_loss = loss_discrete
            self.best_block = self.current_block.cpu().clone()
            self.best_edge_index = self.block_edge_index.cpu().clone()
            self.best_pert_edge_weight = self.block_edge_weight.cpu().clone()

        # Resampling of search space
        if epoch < self.epochs_resampling - 1 and (epoch + 1) % self.resample_period == 0:
            self._resample_random_block(budget)
        elif epoch == self.epochs_resampling - 1:
            # Retrieve best epoch if early stopping is active
            # (not explicitly covered by pseudo code)
            self.current_block = self.best_block.to(self.device)
            self.block_edge_index = self.best_edge_index.to(self.device)
            block_edge_weight = self.best_pert_edge_weight.clone()
            self.block_edge_weight = block_edge_weight.to(self.device)

        scalars['metric'] = metric.item()
        scalars['loss_discrete'] = loss_discrete.item()
        self._write_scalars_to_tb(tb_writer, scalars, epoch)
        return scalars
    
    @staticmethod
    def _write_scalars_to_tb(tb_writer: SummaryWriter, scalars, epoch):
        if tb_writer is not None:
            tb_writer.add_scalar('lr', scalars['lr'], epoch)
            tb_writer.add_scalar('max_abs_grad', scalars['max_abs_grad'], epoch)
            tb_writer.add_scalars(
                'prob_mass',
                {
                    'after_update': scalars['prob_mass_after_update'],
                    'after_projection': scalars['prob_mass_after_projection']
                },
                epoch,
            )
            tb_writer.add_scalars(
                'prob_max',
                {
                    'after_update': scalars['prob_mass_after_update_max'],
                    'after_projection': scalars['prob_mass_after_projection_max']
                },
                epoch,
            )
            tb_writer.add_scalar(
                'num_nonzero_weight_after_projection',
                scalars['prob_mass_after_projection_nonzero_weights'],
                epoch,
            )
            if 'loss_discrete' not in scalars:
                tb_writer.add_scalar('loss_continuous', scalars['loss'], epoch)
            else:
                tb_writer.add_scalar('metric_discrete', scalars['metric'], epoch)
                tb_writer.add_scalars(
                    'loss',
                    {'continuous': scalars['loss'], 'discrete': scalars['loss_discrete']},
                    epoch,
                )

    @torch.no_grad()
    def _close(
        self,
        x: Tensor,
        labels: Tensor,
        budget: int,
        idx_attack: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """Clean up and prepare return argument."""
        # Retrieve best epoch if early stopping is active
        # (not explicitly covered by pseudo code)
        if self.coeffs['with_early_stopping']:
            self.current_block = self.best_block.to(self.device)
            self.block_edge_index = self.best_edge_index.to(self.device)
            self.block_edge_weight = self.best_pert_edge_weight.to(self.device)

        # Sample final discrete graph (Algorithm 1, line 16)
        edge_index, flipped_edges = self._sample_final_edges(x, labels, budget, idx_attack=idx_attack, **kwargs)
        return edge_index, flipped_edges
    
    def _get_forward_data(self, x: Tensor, edge_index: Tensor, edge_weight: None | Tensor, discrete: bool) -> Data:
        # create a data object, clone x, since it gets modified inplace in the forward pass
        data = Data(x=x.clone(), edge_index=edge_index, edge_attr=edge_weight)
        return data

    def _forward(self, x: Tensor, edge_index: None | Tensor, edge_weight: Tensor, discrete: bool, **kwargs) -> Tensor:
        assert (discrete and edge_weight is None) or (not discrete and edge_weight is not None)
        data = self._get_forward_data(x, edge_index, edge_weight, discrete)
        # remove isolated components (if specified in cfg), important for efficient node injection
        data, root_node = remove_isolated_components(data)
        # add the "clean" laplacian info, from which will be perturbed
        if not discrete and cfg.posenc_WLapPE.enable and cfg.attack.SAN.enable_pert_grad:
            self._add_laplacian_info(data)
            # check for repeated eigenvalues:
            data.E_rep_slices_min, data.E_rep_slices_max = get_repeated_eigenvalue_slices(
                data.E_clean, cfg.attack.SAN.eps_repeated_eigenvalue,
            )
        return self.model(data, unmodified=discrete, root_node=root_node, **kwargs)
    
    def _add_laplacian_info(self, data):
        data.lap_clean_edge_index = self.lap_edge_index
        data.lap_clean_edge_attr = self.lap_edge_attr
        data.E_clean = self.E_lap
        data.U_clean = self.U_lap.clone()

    def _forward_and_gradient(
        self,
        x: Tensor,
        labels: Tensor,
        idx_attack: Optional[Tensor] = None,
        retain_graph: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """Forward and update edge weights."""
        self.block_edge_weight.requires_grad = True

        # Retrieve sparse perturbed adjacency matrix `A \oplus p_{t-1}`
        # (Algorithm 1, line 6 / Algorithm 2, line 7)
        edge_index, edge_weight = self._get_modified_adj(
            self.edge_index,
            self.edge_weight,
            self.block_edge_index,
            self.block_edge_weight,
        )

        # Get prediction (Algorithm 1, line 6 / Algorithm 2, line 7)
        prediction = self._forward(x, edge_index, edge_weight, discrete=False, **kwargs)
        # Calculate loss combining all each node
        # (Algorithm 1, line 7 / Algorithm 2, line 8)
        loss = self.loss(prediction, labels, idx_attack)
        # Retrieve gradient towards the current block
        # (Algorithm 1, line 7 / Algorithm 2, line 8)
        gradient = torch.autograd.grad(loss, self.block_edge_weight, retain_graph=retain_graph)[0]
        # Computations outside this do not require gradient
        self.block_edge_weight.requires_grad = False

        return loss, gradient

    def _get_modified_adj(
        self,
        edge_index: Tensor,
        edge_weight: Tensor,
        block_edge_index: Tensor,
        block_edge_weight: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Merges adjacency matrix with current block (incl. weights)."""
        if self.is_undirected:
            block_edge_index, block_edge_weight = to_undirected(
                block_edge_index,
                block_edge_weight,
                num_nodes=self.num_nodes,
                reduce='mean',
            )

        modified_edge_index = torch.cat((edge_index.to(self.device), block_edge_index), dim=-1)
        modified_edge_weight = torch.cat((edge_weight.to(self.device), block_edge_weight))

        modified_edge_index, modified_edge_weight = coalesce(
            modified_edge_index,
            modified_edge_weight,
            num_nodes=self.num_nodes,
            reduce='sum',
        )

        # Allow (soft) removal of edges
        is_edge_in_clean_adj = modified_edge_weight > 1
        modified_edge_weight[is_edge_in_clean_adj] = (2 - modified_edge_weight[is_edge_in_clean_adj])
        
        # remove zero weight edges, not needed in sparse representation
        zero_weight_mask = modified_edge_weight > 0
        modified_edge_weight = modified_edge_weight[zero_weight_mask]
        modified_edge_index = modified_edge_index[:, zero_weight_mask]

        return modified_edge_index, modified_edge_weight

    def _get_discrete_sampled_graph(self, sampled_block_edge_weight: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        returns:
        - edge_index and edge_weight -> define the entire discrete graph
        - discrete_block_edge_weight -> are the discrete values for just the block edges
        """
        sampled_block_discrete = (sampled_block_edge_weight > 0).float()
        edge_index, edge_weight = self._get_modified_adj(
            self.edge_index,
            self.edge_weight,
            self.block_edge_index,
            sampled_block_discrete,
        )
        assert torch.all(edge_weight == 1)
        num_removed_edges = 0
        return edge_index, sampled_block_discrete, num_removed_edges

    def _filter_self_loops_in_block(self, with_weight: bool):
        is_not_sl = self.block_edge_index[0] != self.block_edge_index[1]
        self.current_block = self.current_block[is_not_sl]
        self.block_edge_index = self.block_edge_index[:, is_not_sl]
        if with_weight:
            self.block_edge_weight = self.block_edge_weight[is_not_sl]

    def _sample_random_block(self, budget: int = 0):
        for _ in range(self.coeffs['max_trials_sampling']):
            self.current_block = self.sample_edge_indices(self.block_size)
            self.current_block = torch.unique(self.current_block, sorted=True)
            if self.is_undirected:
                self.block_edge_index = self._linear_to_triu_idx(self.num_nodes, self.current_block)
            else:
                self.block_edge_index = self._linear_to_full_idx(self.num_nodes, self.current_block)
                self._filter_self_loops_in_block(with_weight=False)
            self.block_edge_weight = torch.full(self.current_block.shape, self.coeffs['eps'], device=self.device)

            if cfg.attack.eps_init_noised:
                # remove a bit of eps
                self.block_edge_weight -= (
                    0.2 * self.coeffs['eps'] * torch.rand(self.current_block.shape, device=self.device)
                )

            if self.current_block.size(0) >= budget:
                return
        raise RuntimeError('Sampling random block was not successful. Please decrease `budget`.')

    def _resample_random_block(self, budget: int):
        # Keep at most half of the block (i.e. resample low weights)
        sorted_idx = torch.argsort(self.block_edge_weight)
        keep_above = (self.block_edge_weight <= self.coeffs['eps']).sum().long()
        if keep_above < sorted_idx.size(0) // 2:
            keep_above = sorted_idx.size(0) // 2
        sorted_idx = sorted_idx[keep_above:]

        self.current_block = self.current_block[sorted_idx]
        block_edge_weight_prev = self.block_edge_weight[sorted_idx]
        n_prev = sorted_idx.size(0)

        # Sample until enough edges were drawn
        for i in range(self.coeffs['max_trials_sampling']):
            n_edges_resample = self.block_size - self.current_block.size(0)
            lin_index = self.sample_edge_indices(n_edges_resample)
            current_block = torch.cat((self.current_block, lin_index))
            self.current_block, unique_idx = torch.unique(current_block, sorted=True, return_inverse=True)

            if self.is_undirected:
                self.block_edge_index = self._linear_to_triu_idx(self.num_nodes, self.current_block)
            else:
                self.block_edge_index = self._linear_to_full_idx(self.num_nodes, self.current_block)

            # Merge existing weights with new edge weights
            if i > 0:
                block_edge_weight_prev = self.block_edge_weight
                n_prev = block_edge_weight_prev.size(0)

            self.block_edge_weight = torch.full(self.current_block.shape, self.coeffs['eps'], device=self.device)

            if cfg.attack.eps_init_noised:
                # remove a bit of eps
                self.block_edge_weight -= (
                    0.2 * self.coeffs['eps'] * torch.rand(self.current_block.shape, device=self.device)
                )

            self.block_edge_weight[unique_idx[:n_prev]] = block_edge_weight_prev

            if not self.is_undirected:
                self._filter_self_loops_in_block(with_weight=True)

            if self.current_block.size(0) > budget:
                return
        raise RuntimeError('Sampling random block was not successful. Please decrease `budget`.')

    def _sample_final_edges(
        self,
        x: Tensor,
        labels: Tensor,
        budget: int,
        idx_attack: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        best_metric = float('-Inf')
        best_num_removed_edges = 0
        block_edge_weight = self.block_edge_weight
        block_edge_weight[block_edge_weight <= self.coeffs['eps']] = 0

        for i in range(self.coeffs['max_final_samples']):
            sampled_edges = torch.zeros_like(block_edge_weight)
            if i == 0:
                # In first iteration employ top k heuristic instead of sampling
                sampled_edges_idx = torch.topk(block_edge_weight, budget).indices
                sampled_edges[sampled_edges_idx] = block_edge_weight[sampled_edges_idx]
            else:
                sampled_edges_mask = torch.bernoulli(block_edge_weight).to(bool)
                # TODO: sometimes the next line triggers a CUDA device-side assert error, not sure why...
                # (adv. train. GPS)
                sampled_edges[sampled_edges_mask] = block_edge_weight[sampled_edges_mask]
            
            edge_index, discrete_block_edge_weight, num_removed_edges = self._get_discrete_sampled_graph(sampled_edges)

            if discrete_block_edge_weight.sum() > budget:
                # Allowed budget is exceeded
                continue

            prediction = self._forward(x, edge_index, None, discrete=True, **kwargs)
            metric = self.metric(prediction, labels, idx_attack)

            # Save best sample
            if metric > best_metric:
                best_metric = metric
                self.block_edge_weight = discrete_block_edge_weight.clone().cpu()
                best_num_removed_edges = num_removed_edges

        # Recover best sample
        self.block_edge_weight = self.block_edge_weight.to(self.device)
        flipped_edges = self.block_edge_index[:, self.block_edge_weight > 0]
        edge_index, edge_weight = self._get_modified_adj(
            self.edge_index, self.edge_weight, self.block_edge_index, self.block_edge_weight,
        )
        assert torch.all(edge_weight == 1)
        if self.log and best_num_removed_edges > 0:
            logging.info(f"Removed {num_removed_edges} edges to satisfy constraint.")
        return edge_index, flipped_edges
    
    def _get_learning_rate(self, budget: int, epoch: int) -> float:
        # The learning rate is refined heuristically, s.t. (1) it is
        # independent of the number of perturbations (assuming an undirected
        # adjacency matrix) and (2) to decay learning rate during fine-tuning
        # (i.e. fixed search space).
        lr = (budget / self.num_nodes * self.lr / np.sqrt(max(0, epoch - self.epochs_resampling) + 1))
        return lr

    def _update_edge_weights(self, block_edge_weight: Tensor, gradient: Tensor, lr: float) -> Tensor:
        if self.max_edge_weight_update > 0:
            max_gradient = self.max_edge_weight_update / lr
            gradient = torch.clamp(gradient, -max_gradient, max_gradient)
        return block_edge_weight + lr * gradient

    @staticmethod
    def _project(budget: int, values: Tensor, eps: float = 1e-7) -> Tensor:
        r"""Project :obj:`values`:
        :math:`budget \ge \sum \Pi_{[0, 1]}(\text{values})`.
        """
        if torch.clamp(values, 0, 1).sum() > budget:
            left = (values - 1).min()
            right = values.max()
            miu = PRBCDAttack._bisection(values, left, right, budget)
            values = values - miu
        return torch.clamp(values, min=eps, max=1 - eps)

    @staticmethod
    def _bisection(edge_weights: Tensor, a: float, b: float, n_pert: int, eps=1e-5, max_iter=1e3) -> Tensor:
        """Bisection search for projection."""
        def shift(offset: float):
            return (torch.clamp(edge_weights - offset, 0, 1).sum() - n_pert)

        miu = a
        for _ in range(int(max_iter)):
            miu = (a + b) / 2
            # Check if middle point is root
            if (shift(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (shift(miu) * shift(a) < 0):
                b = miu
            else:
                a = miu
            if ((b - a) <= eps):
                break
        return miu

    @staticmethod
    def _num_possible_edges(n: int, is_undirected: bool) -> int:
        """Determine number of possible edges for graph."""
        if is_undirected:
            return n * (n - 1) // 2
        else:
            return int(n**2)  # We filter self-loops later

    @staticmethod
    def _linear_to_triu_idx(n: int, lin_idx: Tensor) -> Tensor:
        """Linear index to upper triangular matrix without diagonal. This is
        similar to
        https://stackoverflow.com/questions/242711/algorithm-for-index-numbers-of-triangular-matrix-coefficients/28116498#28116498
        with number nodes decremented and col index incremented by one.
        """
        nn = n * (n - 1)
        row_idx = n - 2 - torch.floor(
            torch.sqrt(-8 * lin_idx.double() + 4 * nn - 7) / 2.0 - 0.5
        ).long()
        col_idx = 1 + lin_idx + row_idx - nn // 2 + torch.div(
            (n - row_idx) * (n - row_idx - 1), 2, rounding_mode='floor',
        )
        return torch.stack((row_idx, col_idx))
    
    @staticmethod
    def _triu_to_linear_idx(n: int, triu_idx: Tensor) -> Tensor:
        """Given normal edge indices get the corresponding linear triu idx
         - without diagonal: no i == j
         - upper: no i > j
        triu_idx = [[-row_idx-], [-col_idx-]]
        answer based on 
        https://stackoverflow.com/a/27088560
        """
        i = triu_idx[0, :]
        j = triu_idx[1, :]
        assert (i < j).all()
        linear_idx = (n * (n - 1) // 2) - ((n - i) * (n - i - 1) // 2) + j - i - 1
        return linear_idx

    @staticmethod
    def _linear_to_full_idx(n: int, lin_idx: Tensor) -> Tensor:
        """Linear index to dense matrix including diagonal."""
        row_idx = torch.div(lin_idx, n, rounding_mode='floor')
        col_idx = lin_idx % n
        return torch.stack((row_idx, col_idx))
    
    @staticmethod
    def _full_to_linear_idx(n: int, full_idx: Tensor) -> Tensor:
        """Given normal edge indices get the corresponding linear idx
        full_idx = [[-row_idx-], [-col_idx-]]
        """
        i = full_idx[0, :]
        j = full_idx[1, :]
        linear_idx = n * i + j
        return linear_idx

    @staticmethod
    def _margin_loss(
        score: Tensor,
        labels: Tensor,
        idx_mask: Optional[Tensor] = None,
        reduce: Optional[str] = None,
    ) -> Tensor:
        r"""Margin loss between true score and highest non-target score.

        .. math::
            m = - s_{y} + max_{y' \ne y} s_{y'}

        where :math:`m` is the margin :math:`s` the score and :math:`y` the
        labels.

        Args:
            score (Tensor): Some score (*e.g.*, logits) of shape
                :obj:`[n_elem, dim]`.
            labels (LongTensor): The labels of shape :obj:`[n_elem]`.
            idx_mask (Tensor, optional): To select subset of `score` and
                `labels` of shape :obj:`[n_select]`. Defaults to None.
            reduce (str, optional): if :obj:`mean` the result is aggregated.
                Otherwise, return element wise margin.

        :rtype: (Tensor)
        """
        if idx_mask is not None:
            score = score[idx_mask]
            labels = labels[idx_mask]

        linear_idx = torch.arange(score.size(0), device=score.device)
        true_score = score[linear_idx, labels]

        score = score.clone()
        score[linear_idx, labels] = float('-Inf')
        best_non_target_score = score.amax(dim=-1)

        margin_ = best_non_target_score - true_score
        if reduce is None:
            return margin_
        return margin_.mean()

    @staticmethod
    def _tanh_margin_loss(
        prediction: Tensor,
        labels: Tensor,
        idx_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Calculate tanh margin loss, a node-classification loss that focuses
        on nodes next to decision boundary.

        Args:
            prediction (Tensor): Prediction of shape :obj:`[n_elem, dim]`.
            labels (LongTensor): The labels of shape :obj:`[n_elem]`.
            idx_mask (Tensor, optional): To select subset of `score` and
                `labels` of shape :obj:`[n_select]`. Defaults to None.

        :rtype: (Tensor)
        """
        if cfg.dataset.task_type == "classification_binary":
            m = prediction.exp().log1p()
            log_prob = torch.cat([-m, prediction - m], dim=-1)
        elif cfg.dataset.task_type == "classification":
            log_prob = F.log_softmax(prediction, dim=-1)
        else:
            raise NotImplementedError
        margin_ = GRBCDAttack._margin_loss(log_prob, labels, idx_mask)
        loss = torch.tanh(margin_).mean()
        return loss

    @staticmethod
    def _probability_margin_loss(
        prediction: Tensor,
        labels: Tensor,
        idx_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Calculate probability margin loss, a node-classification loss that
        focuses  on nodes next to decision boundary. See `Are Defenses for
        Graph Neural Networks Robust?
        <https://www.cs.cit.tum.de/daml/are-gnn-defenses-robust>`_ for details.

        Args:
            prediction (Tensor): Prediction of shape :obj:`[n_elem, dim]`.
            labels (LongTensor): The labels of shape :obj:`[n_elem]`.
            idx_mask (Tensor, optional): To select subset of `score` and
                `labels` of shape :obj:`[n_select]`. Defaults to None.

        :rtype: (Tensor)
        """
        prob = F.softmax(prediction, dim=-1)
        margin_ = GRBCDAttack._margin_loss(prob, labels, idx_mask)
        return margin_.mean()

    @staticmethod
    def _masked_cross_entropy(
        log_prob: Tensor,
        labels: Tensor,
        idx_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Calculate masked cross entropy loss, a node-classification loss that
        focuses on nodes next to decision boundary.

        Args:
            log_prob (Tensor): Log probabilities of shape :obj:`[n_elem, dim]`.
            labels (LongTensor): The labels of shape :obj:`[n_elem]`.
            idx_mask (Tensor, optional): To select subset of `score` and
                `labels` of shape :obj:`[n_select]`. Defaults to None.

        :rtype: (Tensor)
        """
        # TODO: take prediction as imput and transform to logprob...
        if idx_mask is not None:
            log_prob = log_prob[idx_mask]
            labels = labels[idx_mask]

        is_correct = log_prob.argmax(-1) == labels
        if is_correct.any():
            log_prob = log_prob[is_correct]
            labels = labels[is_correct]

        return F.nll_loss(log_prob, labels)
    
    @staticmethod
    def _train_loss(
        prediction: Tensor,
        labels: Tensor,
        idx_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Calculate the same loss that is configured for training the model.
        """
        if idx_mask is not None:
            prediction = prediction[idx_mask]
            labels = labels[idx_mask]
        return compute_loss(prediction, labels)[0]
    
    @staticmethod
    def _raw_prediction_loss(
        prediction: Tensor,
        labels: Tensor,
        idx_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Loss for classification (node/ graph)
        
        Calculates the negative raw prediction of the correct class as loss.
        Linear function provides constant gradient everywhere, independent of the
        prediction (if it is good or bad).
        The benefit is that it is never zero -> can get gradients even if the
        prediction is "perfect" and e.g. BCE gives loss==0 wich returns no gradients.
        """
        if idx_mask is not None:
            prediction = prediction[idx_mask]
            labels = labels[idx_mask]

        if cfg.dataset.task_type == "classification_binary":
            invert_mask = labels.to(bool)
            prediction[invert_mask] = -prediction[invert_mask]
        elif cfg.dataset.task_type == "classification":
            num_classes = prediction.size(1)
            labels_mask = F.one_hot(labels, num_classes).to(bool)
            prediction[labels_mask] = -prediction[labels_mask]
        else:
            raise NotImplementedError
        loss = prediction.mean()
        return loss

    @staticmethod
    def _reflected_cross_entropy_loss(
        prediction: Tensor,
        labels: Tensor,
        idx_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Loss for classification (node/ graph)
        
        Reverts the logic of cross entropy:
         - cross entropy is almost linear (constant gradient) when prediction is bad, to promote
           updates to better predictions, but goes to zero when prediction gets better (no need 
           to update weights to further improve this prediction)
         - the 'reflected' cross entropy does the opposite: it is almost linear when the prediction
           is good, but goes to zero when the prediction is really bad.

        BCE with raw logits x: 
         - bce_loss(x, y=1) = -log(sigmoid(x))     = log(1 + exp(-x))
         - bce_loss(x, y=0) = -log(1 - sigmoid(x)) = log(1 + exp( x))
        reflected:
         - loss(x, y=1) = -log(1 + exp( x))
         - loss(x, y=0) = -log(1 + exp(-x))

        multiclass CE with raw logits x:
         - ce_loss(x, y) = -log(softmax(x)[y]) = -log( exp(x[y]) / sum(exp(x)) ) = -logsoftmax(x)[y]
        reflected multiclass CE:
         - loss(x, y) = log( sum(exp(x[i] if i != y) / sum(exp(x)) ) = log(sum(exp(x[i] if i != y)) - log(sum(exp(x)))
                      = log( sum(exp(x[i] / sum(exp(x)) if i != y) )
                      = log( sum(softmax(x)[i] if i != y) )
                      = log( 1 - softmax(x)[y] ) = log(1-softmax(x))[y]

        (reflected, because this new function has the same asymptotes as the original cross 
        entropy but mirrored -> is the positive log likelihood of prediction being a wrong class)
        """
        if idx_mask is not None:
            prediction = prediction[idx_mask]
            labels = labels[idx_mask]

        if cfg.dataset.task_type == "classification_binary":
            invert_mask = ~labels.to(bool)
            prediction[:, invert_mask] = -prediction[:, invert_mask]
            loss = (-torch.log1p(prediction.exp())).mean()
        elif cfg.dataset.task_type == "classification":
            num_classes = prediction.size(1)
            labels_mask = ~F.one_hot(labels, num_classes).to(bool)
            masked_prediction = prediction[labels_mask].reshape((-1, prediction.size(1)-1))
            loss = (torch.logsumexp(masked_prediction, 1) - torch.logsumexp(prediction, 1)).mean()
        else:
            raise NotImplementedError        
        return loss
    
    @staticmethod
    def _neg_accuracy_metric(prediction, labels, idx_mask):
        if idx_mask is not None:
            prediction = prediction[idx_mask]
            labels = labels[idx_mask]
        return -(prediction.argmax(-1) == labels).float().mean()

    def _append_statistics(self, mapping: Dict[str, Any]):
        for key, value in mapping.items():
            self.attack_statistics[key].append(value)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'



class GRBCDAttack(PRBCDAttack):
    r"""The Greedy Randomized Block Coordinate Descent (GRBCD) adversarial
    attack from the `Robustness of Graph Neural Networks at Scale
    <https://www.cs.cit.tum.de/daml/robustness-of-gnns-at-scale>`_ paper.

    GRBCD shares most of the properties and requirements with
    :class:`PRBCDAttack`. It also uses an efficient gradient based approach.
    However, it greedily flips edges based on the gradient towards the
    adjacency matrix.

    .. note::
        For examples of using the GRBCD Attack, see
        `examples/contrib/rbcd_attack.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        contrib/rbcd_attack.py>`_
        for a test time attack (evasion).

    Args:
        model (torch.nn.Module): The GNN module to assess.
        block_size (int): Number of randomly selected elements in the
            adjacency matrix to consider.
        epochs (int, optional): Number of epochs (aborts early if
            :obj:`mode='greedy'` and budget is satisfied) (default: :obj:`125`)
        loss (str or callable, optional): A loss to quantify the "strength" of
            an attack. Note that this function must match the output format of
            :attr:`model`. By default, it is assumed that the task is
            classification and that the model returns raw predictions (*i.e.*,
            no output activation) or uses :obj:`logsoftmax`. Moreover, and the
            number of predictions should match the number of labels passed to
            :attr:`attack`. Either pass Callable or one of: :obj:`'masked'`,
            :obj:`'margin'`, :obj:`'prob_margin'`, :obj:`'tanh_margin'`.
            (default: :obj:`'masked'`)
        is_undirected (bool, optional): If :obj:`True` the graph is
            assumed to be undirected. (default: :obj:`True`)
        log (bool, optional): If set to :obj:`False`, will not log any learning
            progress. (default: :obj:`True`)
    """
    coeffs = {'max_trials_sampling': 20, 'eps': 1e-7}

    def __init__(
        self,
        model: torch.nn.Module,
        block_size: int,
        epochs: int = 125,
        loss = 'masked',
        is_undirected: bool = True,
        log: bool = True,
        **kwargs,
    ):
        super().__init__(model, block_size, epochs, loss=loss,
                         is_undirected=is_undirected, log=log, **kwargs)

    @torch.no_grad()
    def _prepare(self, budget: int) -> List[int]:
        """Prepare attack."""
        self.flipped_edges = self.edge_index.new_empty(2, 0).to(self.device)

        # Determine the number of edges to be flipped in each attach step/epoch
        step_size = budget // self.epochs
        if step_size > 0:
            steps = self.epochs * [step_size]
            for i in range(budget % self.epochs):
                steps[i] += 1
        else:
            steps = [1] * budget

        # Sample initial search space (Algorithm 2, line 3-4)
        self._sample_random_block(step_size)

        return steps

    @torch.no_grad()
    def _update(self, step_size: int, gradient: Tensor, *args,
                **kwargs) -> Dict[str, Any]:
        """Update edge weights given gradient."""
        _, topk_edge_index = torch.topk(gradient, step_size)

        flip_edge_index = self.block_edge_index[:, topk_edge_index]
        flip_edge_weight = torch.ones_like(flip_edge_index[0],
                                           dtype=torch.float32)

        self.flipped_edges = torch.cat((self.flipped_edges, flip_edge_index),
                                       axis=-1)

        if self.is_undirected:
            flip_edge_index, flip_edge_weight = to_undirected(
                flip_edge_index, flip_edge_weight, num_nodes=self.num_nodes,
                reduce='mean')
        edge_index = torch.cat(
            (self.edge_index.to(self.device), flip_edge_index.to(self.device)),
            dim=-1)
        edge_weight = torch.cat((self.edge_weight.to(self.device),
                                 flip_edge_weight.to(self.device)))
        edge_index, edge_weight = coalesce(edge_index, edge_weight,
                                           num_nodes=self.num_nodes,
                                           reduce='sum')

        is_one_mask = torch.isclose(edge_weight, torch.tensor(1.))
        self.edge_index = edge_index[:, is_one_mask]
        self.edge_weight = edge_weight[is_one_mask]
        # self.edge_weight = torch.ones_like(self.edge_weight)
        assert self.edge_index.size(1) == self.edge_weight.size(0)

        # Sample initial search space (Algorithm 2, line 3-4)
        self._sample_random_block(step_size)

        # Return debug information
        scalars = {
            'number_positive_entries_in_gradient': (gradient > 0).sum().item()
        }
        return scalars

    def _close(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        """Clean up and prepare return argument."""
        return self.edge_index, self.flipped_edges
