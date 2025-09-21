import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
from torch_geometric.utils import coalesce, get_laplacian, scatter
from torch_sparse import spmm
from graphgps.transform.lap_eig import (
    get_dense_eigh,
    get_lap_decomp_stats,
    eigvec_normalizer,
    get_repeated_eigenvalue_slices,
)
import numpy as np
import scipy


@register_node_encoder("WLapPE")
class WeightedLapPENodeEncoder(torch.nn.Module):
    """Weighted Laplace Positional Embedding node encoder.

    LapPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with LapPE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        dim_in = cfg.share.dim_in  # Expected original input node features dim

        pecfg = cfg.posenc_WLapPE

        model_type = pecfg.model  # Encoder NN model type for PEs
        if model_type not in ['Transformer', 'DeepSet']:
            raise ValueError(f"Unexpected PE model {model_type}")
        self.model_type = model_type

        dim_pe = pecfg.dim_pe  # Size of Laplace PE embedding
        n_layers = pecfg.layers  # Num. layers in PE encoder model
        n_heads = pecfg.n_heads  # Num. attention heads in Trf PE encoder
        post_n_layers = pecfg.post_layers  # Num. layers to apply after pooling
        self.pass_as_var = pecfg.pass_as_var  # Pass PE also as a separate variable

        if dim_emb - dim_pe < 0: # formerly 1, but you could have zero feature size
            raise ValueError(f"LapPE size {dim_pe} is too large for desired embedding size of {dim_emb}.")

        if expand_x and dim_emb - dim_pe > 0:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x and dim_emb - dim_pe > 0

        activation = nn.ReLU

        if model_type == 'Transformer':
            # Transformer model for LapPE
            # Initial projection of eigenvalue and the node's eigenvector value
            self.linear_A = nn.Linear(2, dim_pe)
            encoder_layer = nn.TransformerEncoderLayer(d_model=dim_pe, nhead=n_heads, batch_first=True)
            self.pe_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        else:
            # DeepSet model for LapPE
            self.linear_A = nn.Linear(2, 2 * dim_pe)
            layers = []
            if n_layers == 1:
                layers.append(activation())
            else:
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        
        self.post_mlp = None
        if post_n_layers > 0:
            # MLP to apply post pooling
            layers = []
            if post_n_layers == 1:
                layers.append(nn.Linear(dim_pe, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(dim_pe, 2 * dim_pe))
                layers.append(activation())
                for _ in range(post_n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.post_mlp = nn.Sequential(*layers)


    def forward(self, batch):
        if batch.get("EigVals") is None:
            add_eig(batch)

        EigVals = batch.EigVals
        EigVecs = batch.EigVecs

        if self.training:
            sign_flip = torch.rand(EigVecs.size(1), device=EigVecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            EigVecs = EigVecs * sign_flip.unsqueeze(0)

        pos_enc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2) # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors) x 2

        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 2
        pos_enc = self.linear_A(pos_enc)  # (Num nodes) x (Num Eigenvectors) x dim_pe

        # PE encoder
        if self.model_type == 'Transformer':
            pos_enc = self.pe_encoder(src=pos_enc, src_key_padding_mask=empty_mask[:, :, 0])
        else:  # DeepSet
            pos_enc = self.pe_encoder(pos_enc)

        # Set masked parts back to zero
        pos_enc = pos_enc * (~empty_mask[:, :, [0]]).float()

        # Sum pooling
        pos_enc = pos_enc.sum(1)  # (Num nodes) x dim_pe

        # MLP post pooling
        if self.post_mlp is not None:
            pos_enc = self.post_mlp(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:
            batch.pe_LapPE = pos_enc
        return batch


def add_eig(batch):
    assert batch.num_graphs == 1, "On the fly getting PE currently only works for single graphs."
    num_nodes = batch.x.size(0)
    pe_cfg = cfg.posenc_WLapPE
    max_freqs = pe_cfg.eigen.max_freqs
    lap_norm_type = pe_cfg.eigen.laplacian_norm
    lap_norm_type = lap_norm_type.lower() if lap_norm_type is not None else None
    if lap_norm_type == 'none':
        lap_norm_type = None
    eigvec_norm = pe_cfg.eigen.eigvec_norm

    attack_mode = batch.get("attack_mode", False) and batch.edge_attr is not None
    pert_approx = attack_mode and cfg.attack.SAN.enable_pert_grad
    diff_pert = attack_mode and cfg.attack.SAN.enable_eig_backprop
    assert not (pert_approx and diff_pert), (
        "Either use the perturbation approximation, or backprop through eigh, not both!" 
    )

    if pert_approx:
        E_pert, U_pert, eigenvalues_true, eigenvectors_true = get_spectral_pert_approx(
            batch, num_nodes, lap_norm_type, max_freqs, eigvec_norm,
        )
    elif diff_pert:
        E_pert, U_pert, eigenvalues_true, eigenvectors_true = get_lap_decomp_diff(
            batch, num_nodes, lap_norm_type, max_freqs, eigvec_norm,
        )
    else:
        # no gradient, simply get the laplacian eigendecomposition
        batch.EigVals, batch.EigVecs = get_lap_decomp_stats(
            batch.edge_index,
            batch.edge_attr,
            num_nodes,
            lap_norm_type,
            max_freqs,
            eigvec_norm,
            pad_too_small=True,
            need_full=False,
            return_lap=False,
        )
        batch.EigVals = batch.EigVals.repeat(num_nodes, 1)[:, :, None]
        return

    if cfg.attack.SAN.match_true_signs:
        # for eigenvectors of the unique eigenvalues that might be flipped in relation to the true ones 
        U_pert = invert_wrong_signs(eigenvectors_true, U_pert)

    if cfg.attack.SAN.set_first_pert_zero:
        eigenvalues = torch.zeros((max_freqs, ), device=batch.x.device)
        eigenvalues[1:] = E_pert[1:]
        E_pert = eigenvalues

    debug = False
    if debug:
        E_error = (E_pert - eigenvalues_true).abs()
        U_sim = (U_pert * eigenvectors_true).sum(0)
        print(f"Min eigval error: {E_error.min().item()}")
        print(f"Avg eigval error: {E_error.mean().item()}")
        print(f"Max eigval error: {E_error.max().item()}")
        print(f"Min eigvec sim.: {U_sim.min().item()}")
        print(f"Avg eigvec sim.: {U_sim.mean().item()}")
        print(f"Max eigvec sim.: {U_sim.max().item()}")

    if cfg.attack.SAN.pert_BPDA:
        # BPDA
        evals = eigenvalues_true + E_pert - E_pert.detach()
        evects = eigenvectors_true + U_pert - U_pert.detach()
    else:
        evals, evects = E_pert, U_pert

    # pad if max_freq > num_nodes
    if num_nodes < max_freqs:
        evals = F.pad(evals, (0, max_freqs - num_nodes), value=float('nan'))
        evects = F.pad(evects, (0, max_freqs - num_nodes), value=float('nan'))
    
    batch.EigVals = evals.repeat(num_nodes, 1)[:, :, None]
    batch.EigVecs = evects


def get_spectral_pert_approx(batch, num_nodes, lap_norm_type, max_freqs, eigvec_norm):
    need_true_eigen = (
        cfg.attack.SAN.match_true_eigenspaces
        or cfg.attack.SAN.match_true_signs
        or cfg.attack.SAN.pert_BPDA
    )

    eigenvalues_true, eigenvectors_true = None, None
    if need_true_eigen:
        out = get_lap_decomp_stats(
            batch.edge_index,
            batch.edge_attr,
            num_nodes,
            lap_norm_type,
            max_freqs,
            eigvec_norm,
            pad_too_small=False,
            need_full=cfg.attack.SAN.match_true_eigenspaces,
            return_lap=True,
            no_grad_lap=False,
        )
        eigenvalues_true, eigenvectors_true, lap_attack_edge_index, lap_attack_edge_attr = out
    else:
        lap_attack_edge_index, lap_attack_edge_attr = get_laplacian(
            batch.edge_index,
            batch.edge_attr,
            lap_norm_type,
            num_nodes=num_nodes,
        )
    
    # get eigen perturbation approximation (function of laplacian perturbation)
    delta_lap_edge_index, delta_lap_edge_attr = coalesce(
        torch.cat((lap_attack_edge_index, batch.lap_clean_edge_index), 1),
        torch.cat((lap_attack_edge_attr, -batch.lap_clean_edge_attr)),
        num_nodes=num_nodes,
        reduce="add",
    )
    zero_mask = delta_lap_edge_attr != 0
    delta_lap_edge_index = delta_lap_edge_index[:, zero_mask]
    delta_lap_edge_attr = delta_lap_edge_attr[zero_mask]

    E_clean = batch.E_clean
    U_clean = batch.U_clean
    P = E_clean[:, None] - E_clean[None, :]
    P[P == 0] = float("inf")
    P_inv = 1 / P
    if batch.E_rep_slices_min.size(0) > 0:
        # handle repeated eigenvalues
        basis_transform_repeated_eigenvectors_(
            U_clean,
            delta_lap_edge_index,
            delta_lap_edge_attr,
            num_nodes,
            batch.E_rep_slices_min,
            batch.E_rep_slices_max,
            P_inv,
        )

    # calculate the eigenperturbation from the formulas of matrix perturbation theory
    Q = U_clean.T @ spmm(delta_lap_edge_index, delta_lap_edge_attr, num_nodes, num_nodes, U_clean)
    E_delta = torch.diag(Q)
    U_delta = - U_clean @ (P_inv * Q)
    E_pert = E_clean + E_delta
    U_pert = U_clean + U_delta

    # sort by smallest eigenvalue
    E_pert, idx = torch.sort(E_pert)
    U_pert = U_pert[:, idx]
    # normalize the pert eigenvectors
    U_pert = eigvec_normalizer(U_pert, E_pert, eigvec_norm)
    
    if cfg.attack.SAN.match_true_eigenspaces:
        # find remaining repeated eigenvalues in E_pert and transform the corresponding
        # eigenvectors in U_pert to be as close as possible to the ones in eigenvectors_true
        U_pert = match_eigenspaces(E_pert, U_pert, eigenvectors_true, max_freqs)
        eigenvalues_true = eigenvalues_true[:max_freqs]
        eigenvectors_true = eigenvectors_true[:, :max_freqs]

    # select only the smallest max_freqs
    E_pert = E_pert[:max_freqs]
    U_pert = U_pert[:, :max_freqs]
    return E_pert, U_pert, eigenvalues_true, eigenvectors_true


@torch.no_grad
def basis_transform_repeated_eigenvectors_(
    U,
    delta_lap_edge_index,
    delta_lap_edge_attr,
    num_nodes,
    slices_min,
    slices_max,
    P_inv,
):
    # find the repeated eigenvalue blocks, set the corresponding eigenvector entries
    for i in range(slices_min.size(0)):
        start, end = slices_min[i].item(), slices_max[i].item()
        U_block = U[:, start:end]  # n x m
        # project L_delta into U basis -- m x m <- ((m x n) x (n x n) x (n x m))
        L_delta_block_p: torch.Tensor
        L_delta_block_p = U_block.T @ spmm(delta_lap_edge_index, delta_lap_edge_attr, num_nodes, num_nodes, U_block)
        # do the eigendecomposition in scipy
        _, U_p = get_dense_eigh(L_delta_block_p.cpu().numpy().astype(np.float64), need_full=True)
        U_p = torch.from_numpy(U_p).float().to(U.device)
        # set the block in U to be equal to the eigenvectors of L_delta_block_p projected back
        U[:, start:end] = U_block @ U_p
        # set the entries in P_inv to zero
        P_inv[start:end, start:end] = 0


def match_eigenspaces(E_pert, U_pert, U_true, max_freq):
    slices_min, slices_max = get_repeated_eigenvalue_slices(
        E_pert, cfg.attack.SAN.eps_repeated_eigenvalue,
    )
    num_repeated = slices_min.size(0)
    if num_repeated == 0:
        return U_pert
    U_pert_transformed = U_pert.clone()

    for i in range(slices_min.size(0)):
        start, end = slices_min[i].item(), slices_max[i].item()
        if start >= max_freq:
            break
        U_pert_space = U_pert[:, start:end]  # n x m
        U_true_space = U_true[:, start:end]  # n x m
        A = torch.linalg.lstsq(U_pert_space.detach(), U_true_space)[0]
        U_pert_transformed[:, start:end] = U_pert_space @ A
        if end >= max_freq:
            break
        
    return U_pert_transformed


def invert_wrong_signs(U_reference: torch.Tensor, U_invert: torch.Tensor):
    # cosine similarity, when negative pointing in different directions
    inverted_sign = (U_reference * U_invert).sum(0) < 0
    flipper = U_reference.new_ones((U_reference.size(1), ))
    flipper[inverted_sign] = -1
    return U_invert * flipper[None, :]


def get_lap_decomp_diff(
    batch,
    num_nodes: int | None,
    lap_norm_type: str | None,
    max_freqs: int,
    eigvec_norm: str,
):
    L_edge_index, L_edge_attr = get_laplacian(batch.edge_index, batch.edge_attr, lap_norm_type, num_nodes=num_nodes)
    L_dense = scatter(
        L_edge_attr,
        num_nodes * L_edge_index[0] + L_edge_index[1],
        dim=0,
        dim_size=num_nodes**2,
        reduce='sum',
    ).reshape((num_nodes, num_nodes))

    E, U = torch.linalg.eigh(L_dense)
    P = get_pert_diff_eigenvects(E, U)
    if P is None:
        E_pert, U_pert = E, U
    else:
        L_pert = L_dense + P
        E_pert, U_pert = torch.linalg.eigh(L_pert)
    E = E.detach()
    U = U.detach()

    E = E[:max_freqs]
    E_pert = E_pert[:max_freqs]
    U = U[:, :max_freqs]
    U_pert = U_pert[:, :max_freqs]

    eigvec_normalizer(U, E, eigvec_norm)
    eigvec_normalizer(U_pert, E_pert, eigvec_norm)
    return E_pert, U_pert, E, U


@torch.no_grad
def get_pert_diff_eigenvects(E, U):
    # eta is the minimal space we would like between two eigenvalues
    eta = cfg.attack.SAN.eig_backprop_separation_pert
    E_diff = torch.diff(E)
    # check if it has "repeated" eigenvalues
    needs_pert = torch.any(E_diff < eta)
    if not needs_pert:
        return None
    
    P = get_ev_pert(E, U, eta)
    return P


@torch.no_grad
def get_ev_pert(E: torch.Tensor, U: torch.Tensor, eta: float):
    # find the offsets to add to the eigenvalues such that they are separated by eta
    n, e_min, e_max = E.size(0), E[0].item(), E[-1].item()
    assert eta < (e_max - e_min) / n, (
        f"Too  many eigenvalues ({n}) in the range {[e_min, e_max]} to separate them with eta={eta}"
    )
    # can be solved as a constrained optimization problem:
    #  -> minimize the offsets, such that all eigenvalues are separated by eta (and the range stays the same)
    offsets = torch.from_numpy(get_offsets(E.detach().cpu().numpy(), eta)).float().to(E.device)
    P_ev = U @ torch.diag(offsets) @ U.T
    return P_ev


def get_offsets(e, eta, gtol=1e-9):
    res = scipy.optimize.minimize(
        lambda x: 0.5 * (x ** 2).sum(),
        get_initial_guess(e),
        method='trust-constr',
        jac=lambda x: x,
        hessp=lambda x, p: p,
        constraints=[get_lin_constraint(e, eta)],
        bounds=get_bounds(e, eta),
        options={'gtol': gtol},
    )
    offsets = res.x
    offsets[np.abs(offsets) < gtol] = 0
    return offsets


def get_bounds(e, eta):
    n = e.size
    eta_range = eta * np.arange(n)
    lb = e[0] - e + eta_range
    lb[-1] = 0
    ub = e[-1] - e - eta_range[::-1]
    ub[0] = 0
    return scipy.optimize.Bounds(lb, ub)


def get_initial_guess(e):
    n = e.size
    return np.linspace(e[0], e[-1], n) - e


def get_lin_constraint(e, eta):
    n = e.size
    e_diff = np.diff(e)
    return scipy.optimize.LinearConstraint(get_A(n), lb=eta-e_diff)


def get_A(n):
    return scipy.sparse.dia_matrix((np.array([[-1], [1]]).repeat(n, 1), np.array([0, 1])), (n-1, n))
