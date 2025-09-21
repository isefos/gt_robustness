import torch
from torch_geometric.graphgym.register import register_edge_encoder


@register_edge_encoder('DummyEdge')
class DummyEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.encoder = torch.nn.Embedding(num_embeddings=1, embedding_dim=emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        assert batch.edge_attr is None
        dummy_attr = batch.edge_index.new_zeros(batch.edge_index.shape[1])
        batch.edge_attr = self.encoder(dummy_attr)
        return batch


@register_edge_encoder('WeightedDummyEdge')
class WeightedDummyEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.encoder = torch.nn.Embedding(num_embeddings=1, embedding_dim=emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        # if there are already 1D (probability) edge attr,
        # therefore save the dummies as different name.
        dummy_attr = batch.edge_index.new_zeros(batch.edge_index.shape[1])
        batch.edge_features = self.encoder(dummy_attr)
        return batch
