import numpy as np
import pandas as pd


def load_univariate_csv(path: str, target_col: str, time_col: str | None = None) -> np.ndarray:
    df = pd.read_csv(path)
    if time_col is not None and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)
    df = df.dropna(subset=[target_col])
    return df[target_col].to_numpy(dtype=np.float64)


def train_val_test_split(series: np.ndarray, train_fraction: float, val_fraction: float, test_fraction: float):
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {total}")
    T = len(series)
    train_end = int(T * train_fraction)
    val_end = train_end + int(T * val_fraction)
    return series[:train_end], series[train_end:val_end], series[val_end:]


def normalize_with_train_stats(train_series, val_series, test_series):
    mean = train_series.mean()
    std = train_series.std() + 1e-8
    return (
        (train_series - mean) / std,
        (val_series - mean) / std,
        (test_series - mean) / std,
        mean,
        std,
    )
