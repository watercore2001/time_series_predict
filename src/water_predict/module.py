from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import MeanSquaredError
__all__ = ["TimePrediction"]
import torch
from torch.optim import AdamW


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
        y_hat = self(batch)
        y = batch["y"]
        self.test_metrics.update(y_hat, y)

    def on_test_epoch_end(self):
        metric_values = self.separate_features_metric(self.test_metrics.compute())
        self.log_dict(metric_values, sync_dist=True)
        self.test_metrics.reset()
