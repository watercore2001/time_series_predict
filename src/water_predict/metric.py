from torchmetrics import R2Score, MeanSquaredError, MeanAbsoluteError, MetricCollection
from water_predict.data.dataset import FEATURE_NAMES


def regression_metrics(num_outputs: int):
    metric_dict = {}
    metric_dict.update({f"Shed_R2": R2Score(num_classes=num_outputs),
                        f"Shed_MAE": MeanAbsoluteError(num_classes=num_outputs),
                        f"Shed_RMSE": MeanSquaredError(num_classes=num_outputs, squared=False)})
    return MetricCollection(metric_dict)


def separate_features_metric(metric_values: dict[str, list[float]]) -> dict[str, float]:
    metric_dict = {}
    for metric_name, features in metric_values.items():
        for feature_id, feature_value in enumerate(features):
            metric_dict[f"{metric_name}_{FEATURE_NAMES[feature_id]}"] = feature_value

    return metric_dict
