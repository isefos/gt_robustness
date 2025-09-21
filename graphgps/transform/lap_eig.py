import torch
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, scatter
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import logging
from numpy.typing import NDArray


def _compute_dense_eigh(A: NDArray, max_freqs: int, need_full: bool, driver: str = "evr"):
    if (not need_full) and (max_freqs < A.shape[0]):
        if driver == "evr":
            E, U = eigh(A, subset_by_index=[0, max_freqs-1], driver=driver)
        else:
            E_full, U_full = eigh(A, driver=driver)
            E, U = E_full[:max_freqs], U_full[:, :max_freqs]
    else:
        E, U = eigh(A, driver=driver)
    return E, U


def get_dense_eigh(A: NDArray, max_freqs: int = 10, need_full: bool = True):
    try:
        try:
            E, U = _compute_dense_eigh(A, max_freqs, need_full, driver="evr")
        except:
            try:
                E, U = _compute_dense_eigh(A, max_freqs, need_full, driver="evd")
            except:
                E, U = _compute_dense_eigh(A, max_freqs, need_full, driver="ev")
    except Exception as e:
        logging.error(f"Could not resolve error during eigendecomposition of the matrix:\n{A.tolist()}")
        raise e
    return E, U


def get_lap_decomp_stats(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    num_nodes: int | None,
    lap_norm_type: str | None,
    max_freqs: int,
    eigvec_norm: str,
    pad_too_small: bool = True,
    need_full: bool = False,
    return_lap: bool = False,
    no_grad_lap: bool = True,
):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.
    """
    if no_grad_lap and edge_attr is not None:
        edge_attr = edge_attr.detach()
    lap_norm_type = lap_norm_type.lower() if lap_norm_type is not None else None
    if lap_norm_type == 'none':
        lap_norm_type = None

    L_edge_index, L_edge_attr = get_laplacian(edge_index, edge_attr, lap_norm_type, num_nodes=num_nodes)
    L = to_scipy_sparse_matrix(L_edge_index, L_edge_attr.detach(), num_nodes)

    E, U = None, None
    if not need_full and (4 * max_freqs) < num_nodes:
        # try sparse
        try:
            E, U = eigsh(L, k=max_freqs, which='SM', return_eigenvectors=True)
        except:
            E = None
    if E is None:
        # do dense calculation
        E, U = get_dense_eigh(L.toarray(), max_freqs, need_full)


    if not need_full:
        assert E.size == max_freqs or E.size == num_nodes

    idx = E.argsort()
    E = E[idx]
    U = U[:, idx]

    evals = torch.from_numpy(E).float().clamp_min(0).to(edge_index.device)
    evals[0] = 0
    evects = torch.from_numpy(U).float().to(edge_index.device)

    # Normalize and pad eigen vectors.
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)

    # Pad if less than max_freqs.
    if num_nodes < max_freqs and pad_too_small:
        evals = F.pad(evals, (0, max_freqs - num_nodes), value=float('nan'))
        evects = F.pad(evects, (0, max_freqs - num_nodes), value=float('nan'))

    if return_lap:
        return evals, evects, L_edge_index, L_edge_attr

    return evals, evects


def eigvec_normalizer(EigVecs, EigVals, normalization):
    """
    Implement different eigenvector normalizations.
    """
    eps=1e-12
    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / torch.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs


def get_repeated_eigenvalue_slices(E, eta):
    E_diff = torch.diff(E)
    # check if it has "repeated" eigenvalues
    if not torch.any(E_diff < eta):
        return torch.tensor([]), torch.tensor([])

    pad = E_diff.new_zeros((1, ), dtype=bool)
    edges = torch.diff(torch.cat((pad, E_diff < eta, pad)).to(dtype=torch.int64))
    slices_min = torch.nonzero(edges == 1).flatten()
    slices_max = torch.nonzero(edges == -1).flatten() + 1
    return slices_min, slices_max
