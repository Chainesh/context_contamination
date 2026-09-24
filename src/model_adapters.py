import numpy as np


class ForecastModel:
    name: str = "base"
    scale: str = "normalized"

    def predict_batch(self, contexts: np.ndarray, horizon: int) -> np.ndarray:
        raise NotImplementedError


class NaiveLastValueModel(ForecastModel):
    name = "naive"
    scale = "normalized"

    def predict_batch(self, contexts, horizon):
        return np.repeat(contexts[:, -1:], horizon, axis=1)


class NaiveSeasonalModel(ForecastModel):
    name = "naive_seasonal"
    scale = "normalized"

    def predict_batch(self, contexts, horizon):
        L = contexts.shape[1]
        if L >= horizon:
            return contexts[:, L - horizon:]
        reps = int(np.ceil(horizon / L))
        return np.tile(contexts, (1, reps))[:, :horizon]


class LinearForecaster(ForecastModel):
    name = "linear"
    scale = "normalized"

    def __init__(self, train_contexts, train_futures, ridge=10.0):
        X = np.hstack([train_contexts, np.ones((len(train_contexts), 1))])
        A = X.T @ X + ridge * np.eye(X.shape[1])
        self.W = np.linalg.solve(A, X.T @ train_futures)

    def predict_batch(self, contexts, horizon):
        X = np.hstack([contexts, np.ones((len(contexts), 1))])
        return (X @ self.W)[:, :horizon]


class TTMAdapter(ForecastModel):
    name = "ttm"
    scale = "normalized"

    def __init__(self, model_path="ibm-granite/granite-timeseries-ttm-r2",
                 context_length=512, prediction_length=96, device="cpu", batch_size=64):
        import torch
        from tsfm_public.toolkit.get_model import get_model

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.model = get_model(model_path, context_length=context_length,
                               prediction_length=prediction_length)
        self.model.to(device).eval()

    def predict_batch(self, contexts, horizon):
        torch = self.torch
        preds = []
        with torch.no_grad():
            for start in range(0, len(contexts), self.batch_size):
                batch = contexts[start:start + self.batch_size]
                x = torch.tensor(batch, dtype=torch.float32, device=self.device).unsqueeze(-1)
                out = self.model(past_values=x)
                y = out.prediction_outputs[..., 0].cpu().numpy()
                preds.append(y[:, :horizon])
        return np.concatenate(preds, axis=0)


class TimesFMAdapter(ForecastModel):
    name = "timesfm"
    scale = "raw"

    def __init__(self, model_name="google/timesfm-1.0-200m-pytorch",
                 context_length=512, horizon=96, device="cpu", batch_size=64):
        import timesfm

        self.batch_size = batch_size
        backend = "gpu" if device == "cuda" else "cpu"
        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend=backend,
                per_core_batch_size=batch_size,
                context_len=context_length,
                horizon_len=horizon,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=model_name),
        )

    def predict_batch(self, contexts, horizon):
        inputs = [row for row in contexts]
        freq = [0] * len(inputs)
        point_forecast, _ = self.model.forecast(inputs, freq=freq)
        return np.asarray(point_forecast)[:, :horizon]

    # def predict_batch(self, contexts, horizon):
    #     preds = []
    #     for start in range(0, len(contexts), self.batch_size):
    #         batch = contexts[start:start + self.batch_size]
    #         point_forecast, _ = self.model.forecast(horizon=horizon, inputs=[row for row in batch])
    #         preds.append(np.asarray(point_forecast)[:, :horizon])
    #     return np.concatenate(preds, axis=0)


class ChronosBoltAdapter(ForecastModel):
    name = "chronos"
    scale = "raw"

    def __init__(self, model_name="amazon/chronos-bolt-base",
                 context_length=512, horizon=96, device="cpu", batch_size=64):
        import torch
        from chronos import BaseChronosPipeline

        self.torch = torch
        self.batch_size = batch_size
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.pipeline = BaseChronosPipeline.from_pretrained(
            model_name, device_map=device, torch_dtype=dtype
        )

    def predict_batch(self, contexts, horizon):
        torch = self.torch
        preds = []
        for start in range(0, len(contexts), self.batch_size):
            batch = contexts[start:start + self.batch_size]
            ctx = [torch.tensor(row, dtype=torch.float32) for row in batch]
            _, mean = self.pipeline.predict_quantiles(
                inputs=ctx, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9]
            )
            preds.append(mean.float().cpu().numpy()[:, :horizon])
        return np.concatenate(preds, axis=0)


class TiRexAdapter(ForecastModel):
    name = "tirex"
    scale = "raw"

    def __init__(self, model_name="NX-AI/TiRex",
                 context_length=512, horizon=96, device="cuda", batch_size=64):
        import torch
        from tirex import load_model

        self.torch = torch
        self.batch_size = batch_size
        self.model = load_model(model_name)

    def predict_batch(self, contexts, horizon):
        torch = self.torch
        preds = []
        for start in range(0, len(contexts), self.batch_size):
            batch = contexts[start:start + self.batch_size]
            x = torch.tensor(batch, dtype=torch.float32)
            quantiles, mean = self.model.forecast(context=x, prediction_length=horizon)
            preds.append(np.asarray(mean.cpu())[:, :horizon])
        return np.concatenate(preds, axis=0)


def get_model(name, cfg, device, batch_size, train_contexts=None, train_futures=None):
    ctx_len = cfg["data"]["context_length"]
    horizon = cfg["data"]["horizon"]

    if name == "naive":
        return NaiveLastValueModel()
    if name == "naive_seasonal":
        return NaiveSeasonalModel()
    if name == "linear":
        if train_contexts is None:
            raise ValueError("linear model needs train_contexts / train_futures")
        return LinearForecaster(train_contexts, train_futures, ridge=cfg["models"].get("linear", {}).get("ridge", 10.0))
    if name == "ttm":
        c = cfg["models"]["ttm"]
        return TTMAdapter(c["model_path"], ctx_len, horizon, device, batch_size)
    if name == "timesfm":
        c = cfg["models"]["timesfm"]
        return TimesFMAdapter(c["model_name"], ctx_len, horizon, device, batch_size)
    if name == "chronos":
        c = cfg["models"]["chronos"]
        return ChronosBoltAdapter(c["model_name"], ctx_len, horizon, device, batch_size)
    if name == "tirex":
        c = cfg["models"]["tirex"]
        return TiRexAdapter(c["model_name"], ctx_len, horizon, device, batch_size)
    raise ValueError(f"unknown model name: {name}")
