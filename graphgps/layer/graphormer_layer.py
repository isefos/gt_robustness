import torch
from torch_geometric.utils import to_dense_batch


class GraphormerLayer(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        attention_dropout: float,
        mlp_dropout: float,
    ):
        """Implementation of the Graphormer layer.
        This layer is based on the implementation at:
        https://github.com/microsoft/Graphormer/tree/v1.0
        Note that this refers to v1 of Graphormer.

        Args:
            embed_dim: The number of hidden dimensions of the model
            num_heads: The number of heads of the Graphormer model
            dropout: Dropout applied after the attention and after the MLP
            attention_dropout: Dropout applied within the attention
            input_dropout: Dropout applied within the MLP
        """
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(
            embed_dim, num_heads, attention_dropout, batch_first=True,
        )
        self.input_norm = torch.nn.LayerNorm(embed_dim)
        self.dropout = torch.nn.Dropout(dropout)

        # We follow the paper in that all hidden dims are
        # equal to the embedding dim
        self.mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(mlp_dropout),
            torch.nn.Linear(embed_dim, embed_dim),
            torch.nn.Dropout(dropout),
        )

    def forward(self, data):
        x = self.input_norm(data.x)
        x, real_nodes = to_dense_batch(x, data.batch)
        mask = ~real_nodes
        key_padding_mask = (torch.zeros_like(mask, dtype=torch.float32).masked_fill_(mask, float("-inf")))
        attn_bias = data.attn_bias if hasattr(data, "attn_bias") else None
        x, _ = self.attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_bias,
            need_weights=False,
        )
        x = x[real_nodes, :]
        x = self.dropout(x) + data.x
        data.x = self.mlp(x) + x
        return data
