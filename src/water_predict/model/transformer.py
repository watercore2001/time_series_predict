from torch import nn
import torch
from torch.nn import MultiheadAttention
from water_predict.model.util.embedding import DataEmbedding


class MLP(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim_ratio: int, dropout: float, act_layer: nn.Module = nn.GELU):
        super().__init__()

        hidden_dim = embedding_dim * hidden_dim_ratio

        self.mlp = nn.Sequential(nn.Linear(embedding_dim, hidden_dim),
                                 act_layer(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, embedding_dim))

    def forward(self, x):
        return self.mlp(x)


class TransformerEncoder(nn.Module):
    def __init__(self, depth: int, d_model: int,
                 heads: int = 8, mlp_ratio: int = 1, dropout: float = 0.2):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MultiheadAttention(embed_dim=d_model, num_heads=heads,
                                   dropout=dropout, batch_first=True),
                MLP(embedding_dim=d_model, hidden_dim_ratio=mlp_ratio, dropout=dropout)
            ]))

        # in paper: Transformers without Tears: Improving the Normalization of Self-Attention
        # In pre-norm residual unit, must append an additional normalization after both encoder and decoder
        # so their outputs are appropriately scaled.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        for attn, mlp in self.layers:
            x = attn(x, x, x)[0] + x
            x = self.norm1(x)
            x = mlp(x) + x
            x = self.norm2(x)

        # layer normalization before return
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, depth: int, d_model: int,
                 heads: int = 8, mlp_ratio: int = 1, dropout: float = 0.2):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MultiheadAttention(embed_dim=d_model, num_heads=heads,
                                                    dropout=dropout, batch_first=True),
                MultiheadAttention(embed_dim=d_model, num_heads=heads,
                                                    dropout=dropout, batch_first=True),

                MLP(embedding_dim=d_model, hidden_dim_ratio=mlp_ratio, dropout=dropout)
            ]))

        # in paper: Transformers without Tears: Improving the Normalization of Self-Attention
        # In pre-norm residual unit, must append an additional normalization after both encoder and decoder
        # so their outputs are appropriately scaled.
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_out):
        for attn1, attn2, mlp in self.layers:
            x = attn1(x, x, x)[0] + x
            x = self.norm1(x)
            x = attn2(x, encoder_out, encoder_out)[0] + x
            x = self.norm2(x)
            x = mlp(x) + x
            x = self.norm3(x)

        # layer normalization before return
        return x


class Transformer(nn.Module):
    def __init__(self, c_in: int, c_out: int, encoder_depth: int, decoder_depth, d_model: int,
                 use_station: bool, use_watershed: bool, use_latlng: bool):
        super().__init__()
        self.embedding1 = DataEmbedding(c_in, d_model,
                                        use_station=use_station, use_watershed=use_watershed, use_latlng=use_latlng)

        self.embedding2 = DataEmbedding(c_in, d_model,
                                        use_station=use_station, use_watershed=use_watershed, use_latlng=use_latlng)

        self.encoder = TransformerEncoder(encoder_depth, d_model)
        self.decoder = TransformerDecoder(decoder_depth, d_model)
        self.linear = nn.Linear(d_model, c_out)

    def forward(self, batch):
        b, y_length, _ = batch["y"].shape
        batch["y"] = torch.cat((batch["x"][:, -y_length:], batch["y"]), dim=1)
        batch["y_week_of_years"] = torch.cat((batch["x_week_of_years"][:, -y_length:], batch["y_week_of_years"]), dim=1)

        x = self.embedding1(batch, flag="x")
        y = self.embedding2(batch, flag="y")

        encoder_out = self.encoder(x)
        result = self.decoder(y, encoder_out)
        result = self.linear(result)

        return result[:, -y_length:,]


