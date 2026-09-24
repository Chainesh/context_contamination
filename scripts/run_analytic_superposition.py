import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import set_seed, load_config
from model_adapters import get_model, ForecastModel

warnings.filterwarnings("ignore")


FUNCTIONS = {
    "t":       lambda t: t,
    "t^2":     lambda t: t ** 2,
    "sin(t)":  lambda t: np.sin(t),
    "cos(t)":  lambda t: np.cos(t),
    "cos(at)": lambda t: np.cos(3.0 * t),
}


def make_function_windows(fn, n_windows, L, H, rng):
    contexts = np.empty((n_windows, L))
    futures = np.empty((n_windows, H))
    for i in range(n_windows):
        t0 = rng.uniform(0, 20)
        step = 0.05
        t = t0 + np.arange(L + H) * step
        vals = fn(t)
        vals = (vals - vals.mean()) / (vals.std() + 1e-8)
        contexts[i] = vals[:L]
        futures[i] = vals[L:L + H]
    return contexts, futures


class AutoARIMAModel(ForecastModel):
    name = "arima"
    scale = "normalized"

    def __init__(self, max_p=5, max_q=5, max_d=2, seasonal=True, m=1):
        from pmdarima import auto_arima
        self._auto_arima = auto_arima
        self.kwargs = dict(max_p=max_p, max_q=max_q, max_d=max_d, seasonal=seasonal, m=m,
                            suppress_warnings=True, error_action="ignore", stepwise=True)

    def predict_batch(self, contexts, horizon):
        preds = np.empty((len(contexts), horizon))
        for i, ctx in enumerate(contexts):
            try:
                fit = self._auto_arima(ctx, **self.kwargs)
                preds[i] = fit.predict(n_periods=horizon)
            except Exception:
                preds[i] = ctx[-1]
        return preds


def predict(model, contexts, horizon, mean, std):
    x = contexts * std + mean if model.scale == "raw" else contexts
    pred = model.predict_batch(x, horizon)
    if model.scale == "raw":
        pred = (pred - mean) / std
    return pred


def superposition_gap(model, x, y, horizon):
    f_x = predict(model, x, horizon, 0.0, 1.0)
    f_y = predict(model, y, horizon, 0.0, 1.0)
    f_xy = predict(model, x + y, horizon, 0.0, 1.0)
    return np.mean(np.abs(f_xy - (f_x + f_y)), axis=1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--models", nargs="+", default=["arima", "ttm", "timesfm", "chronos"])
    p.add_argument("--n-windows", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--arima-max-p", type=int, default=5)
    p.add_argument("--arima-max-q", type=int, default=5)
    p.add_argument("--arima-max-d", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    root = os.path.join(os.path.dirname(__file__), "..")
    cfg = load_config(os.path.join(root, args.config))
    set_seed(cfg["experiment"]["seed"])

    L, H = cfg["data"]["context_length"], cfg["data"]["horizon"]
    rng = np.random.default_rng(cfg["experiment"]["seed"])

    fn_windows = {}
    pair = {}
    for fname, fn in FUNCTIONS.items():
        ctx, fut = make_function_windows(fn, args.n_windows, L, H, rng)
        fn_windows[fname] = ctx
        p = rng.permutation(args.n_windows)
        p[p == np.arange(args.n_windows)] = (p[p == np.arange(args.n_windows)] + 1) % args.n_windows
        pair[fname] = p

    print(f"functions={list(FUNCTIONS.keys())}  n_windows={args.n_windows}  models={args.models}")

    rows = []
    per_model_gap = {}
    for model_name in args.models:
        print(f"evaluating {model_name}")
        if model_name == "arima":
            model = AutoARIMAModel(max_p=args.arima_max_p, max_q=args.arima_max_q, max_d=args.arima_max_d)
        else:
            model = get_model(model_name, cfg, args.device, args.batch_size)

        gaps_by_fn = {}
        for fname in FUNCTIONS:
            x = fn_windows[fname]
            y = fn_windows[fname][pair[fname]]
            gaps_by_fn[fname] = superposition_gap(model, x, y, H)

        avg_gap = np.mean([gaps_by_fn[f].mean() for f in FUNCTIONS])
        per_model_gap[model_name] = gaps_by_fn
        row = {"model": model_name, "avg_gap_5fn": float(avg_gap)}
        for fname in FUNCTIONS:
            row[f"gap_{fname}"] = float(gaps_by_fn[fname].mean())
        rows.append(row)
        del model

    df = pd.DataFrame(rows)
    tables_dir = os.path.join(root, "results", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    df.to_csv(os.path.join(tables_dir, "analytic_superposition.csv"), index=False)

    plots_dir = os.path.join(root, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_gap_histograms(per_model_gap, os.path.join(plots_dir, "analytic_superposition_hist.png"))
    plot_per_function_bars(df, os.path.join(plots_dir, "analytic_superposition_bars.png"))

    print("\nsuperposition gap on analytic functions (avg over 5 functions):")
    print(df.round(4).to_string(index=False))
    best = df.loc[df["avg_gap_5fn"].idxmin(), "model"]
    print(f"\nlowest 5-function average gap: {best}")


def plot_gap_histograms(per_model_gap, save_path):
    models = list(per_model_gap.keys())
    palette = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:brown"]
    all_vals = np.concatenate([np.concatenate(list(per_model_gap[m].values())) for m in models])
    bins = np.linspace(0, np.percentile(all_vals, 99), 30)

    ncols = len(models)
    fig, axes = plt.subplots(1, ncols, figsize=(3.6 * ncols, 4), squeeze=False, sharey=True)
    axes = axes[0]
    for ax, m in zip(axes, models):
        vals = np.concatenate(list(per_model_gap[m].values()))
        c = palette[models.index(m) % len(palette)]
        ax.hist(vals, bins=bins, color=c, alpha=0.8)
        ax.axvline(np.median(vals), color="black", linewidth=1.2, linestyle="--",
                   label=f"median={np.median(vals):.3f}")
        ax.set_title(m)
        ax.set_xlabel("superposition gap")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("windows (all 5 functions pooled)")
    fig.suptitle("Analytic-function superposition gap  |  |f(x+y) - (f(x)+f(y))|  (ARIMA = statistical baseline)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


def plot_per_function_bars(df, save_path):
    fns = [c[4:] for c in df.columns if c.startswith("gap_")]
    models = df["model"].tolist()
    x = np.arange(len(fns))
    width = 0.8 / len(models)
    palette = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:brown"]

    fig, ax = plt.subplots(figsize=(1.4 * len(fns) + 4, 4.5))
    for i, m in enumerate(models):
        vals = [df.loc[df.model == m, f"gap_{f}"].iloc[0] for f in fns]
        ax.bar(x + i * width, vals, width, label=m, color=palette[i % len(palette)])
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(fns)
    ax.set_xlabel("function")
    ax.set_ylabel("mean superposition gap")
    ax.set_title("Superposition gap per function (linear fns should be ~0; t^2 nonlinear)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()