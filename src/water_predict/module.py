from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import R2Score, MeanSquaredError, MeanAbsoluteError, MetricCollection
__all__ = ["TimePrediction"]


class TimePrediction(LightningModule):
    def __init__(self, num_outputs: int, model: nn.Module = None):
        super().__init__()

        self.loss = nn.MSELoss()

        # metrics
        metrics = self.regression_metrics(num_outputs=num_outputs)

        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        # must save all hyperparameters for checkpoint
        self.save_hyperparameters(logger=False)

    @staticmethod
    def regression_metrics(num_outputs: int):
        metric_dict = {"R2": R2Score(num_classes=num_outputs), "MAE": MeanAbsoluteError(num_classes=num_outputs),
                       "RMSE": MeanSquaredError(num_classes=num_outputs, squared=False)}
        return MetricCollection(metric_dict)

    @staticmethod
    def separate_features_metric(metric_values: dict[str, list[float]]) -> dict[str, float]:
        metric_dict = {}
        for metric_name, features in metric_values.items():
            for feature_id, feature_value in enumerate(features):
                metric_dict[f"{metric_name}_{feature_id}"] = feature_value

        return metric_dict

    def training_step(self, batch: dict, batch_index: int):
        y_hat = self(batch)
        y = batch["y"]
        loss = self.loss(y_hat, y)
        self.log(name="train_loss", value=loss, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_index: int):
        y_hat = self(batch)
        y = batch["y"]
        loss = self.loss(y_hat, y)
        self.log(name="val_loss", value=loss, on_epoch=True, sync_dist=True)
        self.val_metrics.update(y_hat, y)

    def on_validation_epoch_end(self) -> None:
        metric_values = self.separate_features_metric(self.val_metrics.compute())
        self.log_dict(metric_values, sync_dist=True)
        self.val_metrics.reset()

    def test_step(self, batch: dict, batch_index: int):
        y_hat = self(batch)
        y = batch["y"]
        self.test_metrics.update(y_hat, y)

    def on_test_epoch_end(self):
        metric_values = self.separate_features_metric(self.test_metrics.compute())
        self.log_dict(metric_values, sync_dist=True)
        self.test_metrics.reset()
