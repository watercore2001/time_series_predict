from torch import nn
import torch
from einops import repeat


class Constant(nn.Module):
    def __init__(self,  c_out: int):
        super().__init__()
        self.result = nn.Parameter(torch.zeros(c_out))

    def forward(self, batch: dict):
        b, y_length, _ = batch["y"].shape

        return repeat(self.result, "c -> b l c", b=b, l=y_length).contiguous()
