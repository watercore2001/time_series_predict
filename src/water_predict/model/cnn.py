from torch import nn
import torch
from water_predict.model.util.embedding import DataEmbedding
from einops import rearrange


class CNN(nn.Module):
    def __init__(self, c_in: int, c_out: int, d_model: int, depth: int,
                 use_station: bool, use_watershed: bool, use_latlng: bool):
        super().__init__()
        self.embedding1 = DataEmbedding(c_in, d_model,
                                        use_station=use_station, use_watershed=use_watershed, use_latlng=use_latlng)

        self.embedding2 = DataEmbedding(c_in, d_model,
                                        use_station=use_station, use_watershed=use_watershed, use_latlng=use_latlng)

        self.conv = nn.Sequential(*[nn.Conv1d(d_model, d_model, kernel_size=3, padding=1) for _ in range(depth)])
        self.linear = nn.Linear(d_model, c_out)

    def forward(self, batch: dict):
        x = self.embedding1(batch, flag="x")
        y = self.embedding2(batch, flag="y")

        _, y_length, _ = y.shape
        result = torch.cat((x, y), dim=1)
        result = rearrange(result, 'b l c -> b c l')
        result = self.conv(result)
        result = rearrange(result, 'b c l -> b l c')
        result = self.linear(result)

        return result[:, -y_length:,]


