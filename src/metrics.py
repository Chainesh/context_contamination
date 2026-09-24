import numpy as np


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def smape(y_true, y_pred, eps=1e-8):
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom))


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "mae": mae(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }


def relative_degradation(metric_at_setting: float, clean_metric: float) -> float:
    return metric_at_setting / clean_metric


def percent_increase(metric_at_setting: float, clean_metric: float) -> float:
    return 100.0 * (metric_at_setting - clean_metric) / clean_metric
