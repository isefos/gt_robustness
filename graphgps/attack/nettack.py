from graphgps.attack.nettack_utils import SparseLocalAttack, OriginalNettack
from torch_sparse import SparseTensor, coalesce
import torch
from torch.nn import Identity


class Nettack(SparseLocalAttack):
    """Wrapper around the implementation of the method proposed in the paper:
    'Adversarial Attacks on Neural Networks for Graph Data'
    by Daniel Zügner, Amir Akbarnejad and Stephan Günnemann,
    published at SIGKDD'18, August 2018, London, UK

    Parameters
    ----------
    adj : torch_sparse.SparseTensor
        [n, n] sparse adjacency matrix.
    X : torch.Tensor
        [n, d] feature matrix.
    labels : torch.Tensor
        Labels vector of shape [n].
    idx_attack : np.ndarray
        Indices of the nodes which are to be attacked [?].
    model : GCN
        Model to be attacked.
    """

    def __init__(
        self,
        **kwargs,
        # adj: Union[SparseTensor, TensorType["n_nodes", "n_nodes"]],
        # attr: TensorType["n_nodes", "n_features"],
        # labels: TensorType["n_nodes"],
        # idx_attack: np.ndarray,
        # model,
        # device: Union[str, int, torch.device],
        # data_device: Union[str, int, torch.device],
        # make_undirected: bool,
        # binary_attr: bool,
        # loss_type: str = 'CE',
    ):
        SparseLocalAttack.__init__(
            self,
            **kwargs,
        )

        assert self.make_undirected, 'Attack only implemented for undirected graphs'

        assert len(self.attacked_model.model.layers) == 2, "Nettack supports only 2 Layer Linear GCN as surrogate model"
        assert isinstance(self.attacked_model.model._modules['activation'], Identity), \
            "Nettack only supports Linear GCN as surrogate model"

        self.sp_adj = self.adj.to_scipy(layout="csr")
        self.sp_attr = SparseTensor.from_dense(self.attr).to_scipy(layout="csr")
        self.nettack = None

    def _attack(self, n_perturbations: int, node_idx: int, direct: bool, n_influencers: int, **kwargs):

        self.nettack = OriginalNettack(self.sp_adj,
                                       self.sp_attr,
                                       self.labels.detach().cpu().numpy(),
                                       self.attacked_model.model.layers[0][0].lin.weight.T.detach().cpu().numpy(),
                                       self.attacked_model.model.layers[1][0].lin.weight.T.detach().cpu().numpy(),
                                       node_idx,
                                       verbose=True)
        self.nettack.reset()
        self.nettack.attack_surrogate(n_perturbations,
                                      perturb_structure=True,
                                      perturb_features=False,
                                      direct=direct,
                                      n_influencers=n_influencers)

        # save to self to get later
        perturbed_idx = self.get_perturbed_edges().T
        self.perturbed_idx = perturbed_idx

        if self.make_undirected:
            perturbed_idx = torch.cat((perturbed_idx, perturbed_idx.flip(0)), dim=-1)

        A_rows, A_cols, A_vals = self.adj.coo()
        A_idx = torch.stack([A_rows, A_cols], dim=0)

        pert_vals = torch.where(
            torch.diag(self.adj[perturbed_idx[0].tolist(), perturbed_idx[1].tolist()].to_dense()) == 0,
            torch.ones_like(perturbed_idx[0]),
            -torch.ones_like(perturbed_idx[0]))

        # sparse addition: A + pert
        A_idx = torch.cat((A_idx, perturbed_idx), dim=-1)
        A_vals = torch.cat((A_vals, pert_vals))
        A_idx, A_vals = coalesce(
            A_idx,
            A_vals,
            m=self.n,
            n=self.n,
            op='sum'
        )

        self.adj_adversary = SparseTensor.from_edge_index(A_idx, A_vals, (self.n, self.n))

    def get_logits(self, model, node_idx: int, perturbed_graph: SparseTensor = None) -> torch.Tensor:
        if perturbed_graph is None:
            perturbed_graph = self.adj

        #if type(model) in BATCHED_PPR_MODELS.__args__:
        #    return model.forward(self.attr, perturbed_graph, ppr_idx=np.array([node_idx]))
        if True:  #else:
            return model.to(self.device)(data=self.attr.to(self.device),
                                         adj=perturbed_graph.to(self.device))[node_idx:node_idx + 1]

    def get_perturbed_edges(self):
        if self.nettack is None:
            return torch.tensor([], device=self.data_device)
        return torch.tensor(self.nettack.structure_perturbations, device=self.data_device)
