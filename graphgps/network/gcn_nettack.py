import collections
from typing import Optional, Union
import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch
from torch_sparse import SparseTensor
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.config import cfg


class ChainableGCNConv(GCNConv):
    """Simple extension to allow the use of `nn.Sequential` with `GCNConv`. The arguments are wrapped as a Tuple/List
    are are expanded for Pytorch Geometric.

    Parameters
    ----------
    See https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#module-torch_geometric.nn.conv.gcn
    """

    def __init__(self, *input, **kwargs):
        super().__init__(*input, **kwargs)

    def forward(self, arguments):
        """Predictions based on the input.

        Parameters
        ----------
        arguments : Sequence[torch.Tensor]
            [x, edge indices] or [x, edge indices, edge weights], by default None

        Returns
        -------
        torch.Tensor
            the output of `GCNConv`.

        Raises
        ------
        NotImplementedError
            if the arguments are not of length 2 or 3
        """
        if len(arguments) == 2:
            x, edge_index = arguments
            edge_weight = None
        elif len(arguments) == 3:
            x, edge_index, edge_weight = arguments
        else:
            raise NotImplementedError("This method is just implemented for two or three arguments")
        embedding = super(ChainableGCNConv, self).forward(x, edge_index, edge_weight=edge_weight)
        #if int(torch_geometric.__version__.split('.')[1]) < 6:
        #    embedding = super(ChainableGCNConv, self).update(embedding)
        return embedding

    def message_and_aggregate(self, adj_t: Union[torch.Tensor, SparseTensor], x: torch.Tensor) -> torch.Tensor:
        return super(ChainableGCNConv, self).message_and_aggregate(adj_t, x)


@register_network('NettackGCN')
class NettackGCN(nn.Module):
    """Two layer GCN implemntation for Nettack surrogate model.
    Parameters
    ----------
    n_features : int
        Number of attributes for each node
        Number of classes for prediction
    n_classes : int
    n_filters : int, optional
        number of dimensions for the hidden units, by default 64
    bias (bool, optional): If set to :obj:`False`, the gcn layers will not learn
            an additive bias. (default: :obj:`True`)
    dropout : int, optional
        Dropout rate, by default 0.5
    do_cache_adj_prep : bool, optional
        If `True` the preoprocessing of the adjacency matrix is chached for training, by default True
    do_normalize_adj_once : bool, optional
        If true the adjacency matrix is normalized only once, by default True
    """

    def __init__(self, dim_in: int, dim_out: int):
        assert cfg.gnn.head == "inductive_node"
        assert cfg.gnn.act == "identity"
        assert cfg.gnn.layer_type == "gcnconvweighted"
        assert cfg.gnn.layers_mp == 2
        #assert cfg.gnn.layers_post_mp == 1
        #assert cfg.gnn.layers_pre_mp == 0
        assert not cfg.gnn.batchnorm
        # exactly two layers
        self.n_filters: list[int] = [cfg.gnn.dim_inner]
        super().__init__()
        self.activation = nn.Identity()
        self.n_features = dim_in
        self.bias = True
        self.n_classes = dim_out
        self.dropout = cfg.gnn.dropout
        self.with_batch_norm = False
        self.layers = self._build_layers()

    def _build_conv_layer(self, in_channels: int, out_channels: int):
        return ChainableGCNConv(
            in_channels=in_channels, out_channels=out_channels, bias=self.bias
        )

    def _build_layers(self):
        filter_dimensions = [self.n_features] + self.n_filters
        modules = nn.ModuleList([
            nn.Sequential(collections.OrderedDict(
                [(f'gcn_{idx}', self._build_conv_layer(in_channels=in_channels, out_channels=out_channels))]
                + ([(f'bn_{idx}', torch.nn.BatchNorm1d(out_channels))] if self.with_batch_norm else [])
                + [(f'activation_{idx}', self.activation),
                   (f'dropout_{idx}', nn.Dropout(p=self.dropout))]
            ))
            for idx, (in_channels, out_channels)
            in enumerate(zip(filter_dimensions[:-1], self.n_filters))
        ])
        idx = len(modules)
        modules.append(nn.Sequential(collections.OrderedDict([
            (f'gcn_{idx}', self._build_conv_layer(in_channels=filter_dimensions[-1], out_channels=self.n_classes)),
        ])))
        return modules

    def forward(self, batch: Batch):
        """
                data: Optional[Union[Data, TensorType["n_nodes", "n_features"]]] = None,
                adj: Optional[Union[SparseTensor,
                                    torch.sparse.FloatTensor,
                                    Tuple[TensorType[2, "nnz"], TensorType["nnz"]],
                                    TensorType["n_nodes", "n_nodes"]]] = None,
                attr_idx: Optional[TensorType["n_nodes", "n_features"]] = None,
                edge_idx: Optional[TensorType[2, "nnz"]] = None,
                edge_weight: Optional[TensorType["nnz"]] = None,
                n: Optional[int] = None,
                d: Optional[int] = None) -> TensorType["n_nodes", "n_classes"]:
        """
        x, edge_idx, edge_weight = batch.x, batch.edge_index, batch.edge_attr

        device = next(self.parameters()).device
        if x.device != device:
            x = x.to(device)
        if edge_idx.device != device:
            edge_idx = edge_idx.to(device)
        if edge_weight is not None and edge_weight.device != device:
            edge_weight = edge_weight.to(device)

        # Enforce that the input is contiguous
        x, edge_idx, edge_weight = self._ensure_contiguousness(x, edge_idx, edge_weight)

        for layer in self.layers:
            x = layer((x, edge_idx, edge_weight))
        batch.x = x

        return x, batch.y

    def _ensure_contiguousness(self,
                               x: torch.Tensor,
                               edge_idx: Union[torch.Tensor, SparseTensor],
                               edge_weight: Optional[torch.Tensor]):
        if not x.is_sparse:
            x = x.contiguous()
        if hasattr(edge_idx, 'contiguous'):
            edge_idx = edge_idx.contiguous()
        if edge_weight is not None:
            edge_weight = edge_weight.contiguous()
        return x, edge_idx, edge_weight
