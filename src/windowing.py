import numpy as np


def make_windows(series: np.ndarray, context_length: int, horizon: int, stride: int = 1, max_windows: int | None = None):
    T = len(series)
    if T < context_length + horizon:
        raise ValueError(f"series length {T} too short for context_length={context_length} + horizon={horizon}")

    contexts, futures = [], []
    for start in range(0, T - context_length - horizon + 1, stride):
        contexts.append(series[start:start + context_length])
        futures.append(series[start + context_length:start + context_length + horizon])
        if max_windows is not None and len(contexts) >= max_windows:
            break

    return np.stack(contexts), np.stack(futures)
