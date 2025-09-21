from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix,
    to_undirected,
    to_dense_adj,
    scatter,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from graphgps.encoder.graphormer_encoder import graphormer_pre_processing
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
from graphgps.transform.rrwp import add_full_rrwp
from graphgps.transform.lap_eig import get_lap_decomp_stats


def compute_posenc_stats(data, pe_types, is_undirected, cfg):
    """Precompute positional encodings for the given graph.

    Supported PE statistics to precompute, selected by `pe_types`:
    'LapPE': Laplacian eigen-decomposition.
    'RWSE': Random walk landing probabilities (diagonals of RW matrices).
    'HKfullPE': Full heat kernels and their diagonals. (NOT IMPLEMENTED)
    'HKdiagSE': Diagonals of heat kernel diffusion.
    'ElstaticSE': Kernel based on the electrostatic interaction between nodes.
    'Graphormer': Computes spatial types and optionally edges along shortest paths.
    'RRWP': Relative Random Walk Probabilities PE (for GRIT)

    Args:
        data: PyG graph
        pe_types: Positional encoding types to precompute statistics for.
            This can also be a combination, e.g. 'eigen+rw_landing'
        is_undirected: True if the graph is expected to be undirected
        cfg: Main configuration node

    Returns:
        Extended PyG Data object.
    """
    # Verify PE types.
    for t in pe_types:
        if t not in [
            'LapPE', 'EquivStableLapPE', 'SignNet', 'RWSE', 'HKdiagSE', 'HKfullPE', 'ElstaticSE', 'GraphormerBias',
            'RRWP', 'WLapPE',
        ]:
            raise ValueError(f"Unexpected PE stats selection {t} in {pe_types}")

    # Basic preprocessing of the input graph.
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    if is_undirected:
        undir_edge_index = data.edge_index
    else:
        undir_edge_index = to_undirected(data.edge_index)

    # Eigen values and vectors, Eigen-decomposition can be reused for Heat kernels.
    eigenvalues, eigenvectors, lap_norm = None, None, None
    eigPEs = set(('LapPE', 'WLapPE', 'EquivStableLapPE', 'SignNet'))
    eigPE = eigPEs.intersection(pe_types)
    assert len(eigPE) <= 1, "Selection of only one eigen-decomposition PE type currently supported."
    if eigPE:
        eigPE = eigPE.pop()
        pe_cfg = cfg.get(f"posenc_{eigPE}")
        lap_norm = pe_cfg.eigen.laplacian_norm
        assert is_undirected, (
            "Lap. Eig. decomposition currently requires undirected graphs (transform dataset to undirected)."
        )
        num_nodes = data.x.size(0)
        eigenvalues, eigenvectors = get_lap_decomp_stats(
            data.edge_index,
            data.edge_attr,
            num_nodes,
            pe_cfg.eigen.laplacian_norm,
            max_freqs=pe_cfg.eigen.max_freqs,
            eigvec_norm=pe_cfg.eigen.eigvec_norm,
        )
        eigenvalues = eigenvalues.repeat(num_nodes, 1)[:, :, None]
        if eigPE == 'SignNet':
            data.eigvals_sn, data.eigvecs_sn = eigenvalues, eigenvectors
        else:
            data.EigVals, data.EigVecs = eigenvalues, eigenvectors

    # Random Walks.
    if 'RWSE' in pe_types:
        kernel_param = cfg.posenc_RWSE.kernel
        if len(kernel_param.times) == 0:
            raise ValueError("List of kernel times required for RWSE")
        rw_landing = get_rw_landing_probs(
            ksteps=kernel_param.times,
            edge_index=data.edge_index,
            num_nodes=N,
        )
        data.pestat_RWSE = rw_landing

    # Heat Kernels.
    if 'HKdiagSE' in pe_types or 'HKfullPE' in pe_types:
        # Get the eigenvalues and eigenvectors of the regular Laplacian,
        # if they have not yet been computed for 'eigen'.
        if lap_norm is not None or eigenvalues is None or eigenvectors is None:
            L_heat = to_scipy_sparse_matrix(
                *get_laplacian(undir_edge_index, normalization=None, num_nodes=N)
            )
            evals_heat, evects_heat = np.linalg.eigh(L_heat.toarray())
        else:
            evals_heat, evects_heat = eigenvalues, eigenvectors
        evals_heat = torch.from_numpy(evals_heat)
        evects_heat = torch.from_numpy(evects_heat)

        # Get the full heat kernels.
        if 'HKfullPE' in pe_types:
            # The heat kernels can't be stored in the Data object without
            # additional padding because in PyG's collation of the graphs the
            # sizes of tensors must match except in dimension 0. Do this when
            # the full heat kernels are actually used downstream by an Encoder.
            raise NotImplementedError()
            # heat_kernels, hk_diag = get_heat_kernels(evects_heat, evals_heat,
            #                                   kernel_times=kernel_param.times)
            # data.pestat_HKdiagSE = hk_diag
        # Get heat kernel diagonals in more efficient way.
        if 'HKdiagSE' in pe_types:
            kernel_param = cfg.posenc_HKdiagSE.kernel
            if len(kernel_param.times) == 0:
                raise ValueError("Diffusion times are required for heat kernel")
            hk_diag = get_heat_kernels_diag(
                evects_heat,
                evals_heat,
                kernel_times=kernel_param.times,
                space_dim=0,
            )
            data.pestat_HKdiagSE = hk_diag

    # Electrostatic interaction inspired kernel.
    if 'ElstaticSE' in pe_types:
        elstatic = get_electrostatic_function_encoding(undir_edge_index, N)
        data.pestat_ElstaticSE = elstatic

    # Graphormer PE: shortest paths and degrees
    if 'GraphormerBias' in pe_types:
        with torch.no_grad():
            data = graphormer_pre_processing(data, is_undirected)
    
    # GRIT PE (Relative Random Walk Probabilities)
    if 'RRWP' in pe_types:
        with torch.no_grad():
            data = add_full_rrwp(data, walk_length=cfg.posenc_RRWP.ksteps)

    return data


def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter(edge_weight, source, dim=0, dim_size=num_nodes, reduce='sum')  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing


def get_heat_kernels_diag(evects, evals, kernel_times=[], space_dim=0):
    """Compute Heat kernel diagonal.

    This is a continuous function that represents a Gaussian in the Euclidean
    space, and is the solution to the diffusion equation.
    The random-walk diagonal should converge to this.

    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the diffusion diagonal by a factor `t^(space_dim/2)`. In
            euclidean space, this correction means that the height of the
            gaussian stays constant across time, if `space_dim` is the dimension
            of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    heat_kernels_diag = []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels diagonal only for each time
        eigvec_mul = evects ** 2
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j} * phi_{i, j})
            this_kernel = torch.sum(torch.exp(-t * evals) * eigvec_mul,
                                    dim=0, keepdim=False)

            # Multiply by `t` to stabilize the values, since the gaussian height
            # is proportional to `1/t`
            heat_kernels_diag.append(this_kernel * (t ** (space_dim / 2)))
        heat_kernels_diag = torch.stack(heat_kernels_diag, dim=0).transpose(0, 1)

    return heat_kernels_diag


def get_heat_kernels(evects, evals, kernel_times=[]):
    """Compute full Heat diffusion kernels.

    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
    """
    heat_kernels, rw_landing = [], []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1).unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels for each time
        eigvec_mul = (evects.unsqueeze(2) * evects.unsqueeze(1))  # (phi_{i, j1, ...} * phi_{i, ..., j2})
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j1, ...} * phi_{i, ..., j2})
            heat_kernels.append(
                torch.sum(torch.exp(-t * evals) * eigvec_mul,
                          dim=0, keepdim=False)
            )

        heat_kernels = torch.stack(heat_kernels, dim=0)  # (Num kernel times) x (Num nodes) x (Num nodes)

        # Take the diagonal of each heat kernel,
        # i.e. the landing probability of each of the random walks
        rw_landing = torch.diagonal(heat_kernels, dim1=-2, dim2=-1).transpose(0, 1)  # (Num nodes) x (Num kernel times)

    return heat_kernels, rw_landing


def get_electrostatic_function_encoding(edge_index, num_nodes):
    """Kernel based on the electrostatic interaction between nodes.
    """
    L = to_scipy_sparse_matrix(
        *get_laplacian(edge_index, normalization=None, num_nodes=num_nodes)
    ).todense()
    L = torch.as_tensor(L)
    Dinv = torch.eye(L.shape[0]) * (L.diag() ** -1)
    A = deepcopy(L).abs()
    A.fill_diagonal_(0)
    DinvA = Dinv.matmul(A)

    electrostatic = torch.pinverse(L)
    electrostatic = electrostatic - electrostatic.diag()
    green_encoding = torch.stack([
        electrostatic.min(dim=0)[0],  # Min of Vi -> j
        electrostatic.max(dim=0)[0],  # Max of Vi -> j
        electrostatic.mean(dim=0),  # Mean of Vi -> j
        electrostatic.std(dim=0),  # Std of Vi -> j
        electrostatic.min(dim=1)[0],  # Min of Vj -> i
        electrostatic.max(dim=0)[0],  # Max of Vj -> i
        electrostatic.mean(dim=1),  # Mean of Vj -> i
        electrostatic.std(dim=1),  # Std of Vj -> i
        (DinvA * electrostatic).sum(dim=0),  # Mean of interaction on direct neighbour
        (DinvA * electrostatic).sum(dim=1),  # Mean of interaction from direct neighbour
    ], dim=1)

    return green_encoding


class ComputePosencStat(BaseTransform):
    def __init__(self, pe_types, is_undirected, cfg):
        self.pe_types = pe_types
        self.is_undirected = is_undirected
        self.cfg = cfg

    def __call__(self, data: Data) -> Data:
        data = compute_posenc_stats(
            data,
            pe_types=self.pe_types,
            is_undirected=self.is_undirected,
            cfg=self.cfg,
        )
        return data