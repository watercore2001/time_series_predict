import torch
import torch.nn as nn
from typing import Literal
import math
from einops import rearrange
from water_predict.data.dataset import STATION_NUM, WATERSHED_NUM, WEEK_NUM


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 1, max_len, d_model
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 1, l, d_model
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.embed = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                               kernel_size=3, padding=padding)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = rearrange(x, 'b l c -> b c l')
        x = self.embed(x)
        x = rearrange(x, 'b c l -> b l c')
        # b l d_model
        return x


class DataEmbedding(nn.Module):

    def __init__(self, c_in: int, d_model: int,
                 use_station: bool, use_watershed: bool, use_latlng: bool,
                 dropout=0.2):
        super().__init__()
        self.use_station = use_station
        self.use_watershed = use_watershed
        self.use_latlng = use_latlng

        self.token_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.week_embedding = nn.Embedding(WEEK_NUM, d_model)

        if self.use_station:
            self.station_embedding = nn.Embedding(STATION_NUM, d_model)
        if self.use_watershed:
            self.watershed_embedding = nn.Embedding(WATERSHED_NUM, d_model)
        if self.use_latlng:
            self.lat_lng_embedding = nn.Linear(in_features=2, out_features=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, batch, flag: Literal["x", "y"]):
        """
        :param batch:
            {"x": x_feature, "x_week_of_years": x_week_of_years,
            "y": y_feature, "y_week_of_years": y_week_of_years,
            "station_id": self.station_id, "watershed_id": self.watershed_id,
            "lat_lng": self.lat_lon}
        :param flag:
        :return:
        """
        if flag == "x":
            result = self.token_embedding(batch["x"])
            result += self.position_embedding(batch["x"])
            result += self.week_embedding(batch["x_week_of_years"])
        elif flag == "y":
            # use 0 as input
            result = self.token_embedding(torch.zeros_like(batch["y"]))
            result += self.position_embedding(batch["y"]).float()
            result += self.week_embedding(batch["y_week_of_years"])
        else:
            raise ValueError

        if self.use_station:
            result += self.station_embedding(batch["station_id"])[:, None, :]
        if self.use_watershed:
            result += self.watershed_embedding(batch["watershed_id"])[:, None, :]
        if self.use_latlng:
            result += self.lat_lng_embedding(batch["lat_lng"])[:, None, :]

        return self.dropout(result)
