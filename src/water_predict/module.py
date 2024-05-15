import os

from pytorch_lightning import LightningModule
from torch import nn

import torch
from torch.optim import AdamW
from einops import rearrange
import numpy as np
import pandas as pd
from torchmetrics import R2Score, MeanSquaredError, MeanAbsoluteError, MetricCollection
from water_predict.data.dataset import WEEK_OF_YEAR, FEATURE_NAMES

__all__ = ["TimePrediction"]

PREDICT_LENGTH = 32


class TimePrediction(LightningModule):
    def __init__(self, model: nn.Module, output_folder: str):
        super().__init__()
        self.output_folder = output_folder

        self.loss = nn.MSELoss()
        self.model = model

        # metrics
        metric = self.regression_metrics()
        self.val_metric = metric.clone(prefix="val_")
        self.test_metric = metric.clone(prefix="test_")

        # must save all hyperparameters for checkpoint
        self.save_hyperparameters(logger=False)

    @staticmethod
    def regression_metrics():
        metric_dict = {}
        metric_dict.update({f"R2": R2Score(),
                            f"MAE": MeanAbsoluteError(),
                            f"MSE": MeanSquaredError()})
        return MetricCollection(metric_dict)

    def forward(self, batch: dict):
        return self.model(batch)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-5)

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

        y_hat = rearrange(y_hat, "b l c -> (b l c)")
        y = rearrange(y, "b l c -> (b l c)")
        output = self.val_metric(y_hat, y)
        self.log_dict(output)

    def test_step(self, batch: dict, batch_index: int):
        y = batch["y"].clone()
        batch["y"] = torch.zeros_like(batch["y"])

        y_hat = self(batch)

        y_hat = rearrange(y_hat, "b l c -> (b l c)")
        y = rearrange(y, "b l c -> (b l c)")
        output = self.test_metric(y_hat, y)
        self.log_dict(output)


    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        output = torch.zeros_like(batch["feature"])
        output[:, :PREDICT_LENGTH] = batch["feature"][:, :PREDICT_LENGTH]
        b, length, c = output.shape

        for i in range(PREDICT_LENGTH, length):
            batch["x"] = batch["feature"][:, i-PREDICT_LENGTH:i]
            batch["x_week_of_years"] = batch["week_of_years"][:, i-PREDICT_LENGTH:i]
            batch["y"] = torch.zeros(b, 1, c).to(batch["x"].device)
            batch["y_week_of_years"] = batch["week_of_years"][:, i].reshape(-1, 1)
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
        path = os.path.join(self.output_folder, f"{watershed_id:d}watershed/{station_id:03d}station.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

