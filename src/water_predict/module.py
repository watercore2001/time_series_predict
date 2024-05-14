import os

from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import MeanSquaredError
__all__ = ["TimePrediction"]
import torch
from torch.optim import AdamW
from einops import rearrange
import numpy as np
import pandas as pd

from water_predict.data.dataset import WEEK_OF_YEAR, FEATURE_NAMES
PREDICT_LENGTH = 32

class TimePrediction(LightningModule):
    def __init__(self, model: nn.Module = None, lr: float = 1e-5):
        super().__init__()

        self.loss = nn.MSELoss()
        self.model = model
        self.lr = lr

        # metrics
        self.RMSE = MeanSquaredError(squared=False)

        # must save all hyperparameters for checkpoint
        self.save_hyperparameters(logger=False)

    def forward(self, batch: dict):
        return self.model(batch)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr)

    def training_step(self, batch: dict, batch_index: int):
        y = batch["y"].clone()
        batch["y"] = torch.zeros_like(batch["y"])

        y_hat = self(batch)
        loss = self.loss(y_hat, y)
        self.log(name="train_loss", value=loss, on_step=True)
        return loss

    def validation_step(self, batch: dict, batch_index: int):
        y = batch["y"].clone()
        batch["y"] = torch.zeros_like(batch["y"])

        y_hat = self(batch)
        loss = self.loss(y_hat, y)
        self.log(name="val_loss", value=loss, on_epoch=True)

        self.RMSE(y_hat, y)
        self.log("val_RMSE", self.RMSE, on_epoch=True)

    def test_step(self, batch: dict, batch_index: int):
        output = torch.zeros_like(batch["feature"])
        output[:, :PREDICT_LENGTH] = batch["feature"][:, :PREDICT_LENGTH]
        b, length, c = output.shape

        for i in range(PREDICT_LENGTH, length):
            batch["x"] = batch["feature"][:, i-PREDICT_LENGTH:i]
            batch["x_week_of_years"] = batch["week_of_years"][:, i-PREDICT_LENGTH:i]
            batch["y"] = torch.zeros(b, 1, c)
            batch["y_week_of_years"] = batch["week_of_years"][:, i].reshape(-1,1)
            y_hat = self(batch)
            y_hat = rearrange(y_hat, "b 1 c -> b c")
            output[:, i] = y_hat

        # save
        feature_array = output[0].cpu().numpy()
        week_of_years_array = batch["week_of_years"][0].cpu().numpy()
        week_of_years_array = np.expand_dims(week_of_years_array, axis=1)

        combined_array = np.hstack((week_of_years_array, feature_array))
        df = pd.DataFrame(combined_array, columns=[WEEK_OF_YEAR]+FEATURE_NAMES)
        station_id = int(batch["station_id"][0].cpu())
        watershed_id = int(batch["watershed_id"][0].cpu())
        path = f"output/{watershed_id:d}watershed/{station_id:03d}station.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

