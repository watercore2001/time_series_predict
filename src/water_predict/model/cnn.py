from torch import nn
import torch
from water_predict.model.util.embedding import DataEmbedding


class CNN(nn.Module):
    def __init__(self, c_in: int, c_out: int, d_model: int, depth: int,
                 use_station: bool, use_watershed: bool, use_latlng: bool):
        super().__init__()
        self.embedding = DataEmbedding(c_in, d_model,
                                       use_station=use_station, use_watershed=use_watershed, use_latlng=use_latlng)

        self.conv = nn.Sequential(*[nn.Conv1d(d_model, d_model, kernel_size=3, padding=1) for _ in range(depth)])
        self.linear = nn.Linear(d_model, c_out)

    def forward(self, batch: dict):
        x = self.embedding(batch, flag="x")
        y = self.embedding(batch, flag="y")
        _, y_length, _ = y.shape
        result = torch.cat((x, y), dim=1)
        result = self.conv(result)
        result = self.linear(result)

        return result[:, -y_length:,]


