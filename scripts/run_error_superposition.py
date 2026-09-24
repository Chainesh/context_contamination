import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import set_seed, load_config
from data_loading import load_univariate_csv, train_val_test_split, normalize_with_train_stats
from windowing import make_windows
from model_adapters import get_model


def predict(model, contexts, horizon, mean, std):
    x = contexts * std + mean if model.scale == "raw" else contexts
    pred = model.predict_batch(x, horizon)
    if model.scale == "raw":
        pred = (pred - mean) / std
    return pred


def pair_indices(n, rng):
    partner = np.empty(n, dtype=np.int64)
    for i in range(n):
        j = rng.integers(0, n - 1)
        partner[i] = j if j < i else j + 1
    return partner


def error_superposition_test(model, contexts, futures, horizon, mean, std, partner):
    x, y = contexts, contexts[partner]
    Tx, Ty = futures, futures[partner]

    f_x = predict(model, x, horizon, mean, std)
    f_y = predict(model, y, horizon, mean, std)
    f_xy = predict(model, x + y, horizon, mean, std)

    # error against the linear-equivalent target Tx + Ty
    e_x = f_x - Tx
    e_y = f_y - Ty
    e_xy = f_xy - (Tx + Ty)

    # the two ways to write the error-superposition gap (identical by construction)
    error_gap = np.mean(np.abs(e_xy - (e_x + e_y)), axis=1)
    forecast_gap = np.mean(np.abs(f_xy - (f_x + f_y)), axis=1)

    scale = np.mean(np.abs(f_xy), axis=1) + 1e-8
    return {
        "f_x": f_x, "f_y": f_y, "f_xy": f_xy, "Tx": Tx, "Ty": Ty,
        "error_gap": error_gap, "forecast_gap": forecast_gap,
        "relative_gap": error_gap / scale,
        "cancellation_residual": np.abs(error_gap - forecast_gap),
    }


def multiplicative_superposition_test(model, contexts, futures, horizon, mean, std, partner):
    # NOTE: unlike the additive case, an "error vs Tx*Ty" framing does NOT cleanly
    # reduce to a forecast-space comparison here (verified symbolically: even when
    # f(x*y) = f(x)*f(y) exactly, e(x*y) - e(x)*e(y) leaves a nonzero residual unless
    # the individual forecasts are also error-free). So this test compares directly
    # in forecast space, which is the well-defined multiplicative analogue.
    x, y = contexts, contexts[partner]

    f_x = predict(model, x, horizon, mean, std)
    f_y = predict(model, y, horizon, mean, std)
    f_xy = predict(model, x * y, horizon, mean, std)

    gap = np.mean(np.abs(f_xy - (f_x * f_y)), axis=1)
    scale = np.mean(np.abs(f_xy), axis=1) + 1e-8
    return {
        "f_x": f_x, "f_y": f_y, "f_xy": f_xy,
        "gap": gap, "relative_gap": gap / scale,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--models", nargs="+", default=["ttm", "timesfm", "chronos"])
    p.add_argument("--max-windows", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    root = os.path.join(os.path.dirname(__file__), "..")
    cfg = load_config(os.path.join(root, args.config))
    set_seed(cfg["experiment"]["seed"])

    series = load_univariate_csv(
        os.path.join(root, cfg["data"]["path"]), cfg["data"]["target_col"], cfg["data"]["time_col"]
    )
    train, val, test = train_val_test_split(
        series, cfg["data"]["train_fraction"], cfg["data"]["val_fraction"], cfg["data"]["test_fraction"]
    )
    train_n, val_n, test_n, mean, std = normalize_with_train_stats(train, val, test)

    L, H = cfg["data"]["context_length"], cfg["data"]["horizon"]
    contexts, futures = make_windows(test_n, L, H, stride=cfg["data"]["stride"], max_windows=args.max_windows)
    train_contexts, train_futures = make_windows(
        train_n, L, H, stride=cfg["data"].get("train_stride", 48), max_windows=None
    )

    rng = np.random.default_rng(cfg["experiment"]["seed"])
    partner = pair_indices(len(contexts), rng)
    print(f"windows={len(contexts)}  target = Tx + Ty (linear-equivalent)  models={args.models}")

    rows = []
    store = {}
    mult_store = {}
    for model_name in args.models:
        print(f"evaluating {model_name}")
        model = get_model(model_name, cfg, args.device, args.batch_size,
                          train_contexts=train_contexts, train_futures=train_futures)
        r = error_superposition_test(model, contexts, futures, H, mean, std, partner)
        rm = multiplicative_superposition_test(model, contexts, futures, H, mean, std, partner)
        store[model_name] = r
        mult_store[model_name] = rm
        rows.append({
            "model": model_name,
            "error_gap_mean": float(r["error_gap"].mean()),
            "error_gap_median": float(np.median(r["error_gap"])),
            "relative_gap_mean": float(r["relative_gap"].mean()),
            "forecast_gap_mean": float(r["forecast_gap"].mean()),
            "cancellation_residual_max": float(r["cancellation_residual"].max()),
            "multiplicative_gap_mean": float(rm["gap"].mean()),
            "multiplicative_gap_median": float(np.median(rm["gap"])),
            "multiplicative_relgap_mean": float(rm["relative_gap"].mean()),
        })
        del model

    df = pd.DataFrame(rows)
    tables_dir = os.path.join(root, "results", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    df.to_csv(os.path.join(tables_dir, "error_superposition.csv"), index=False)

    plots_dir = os.path.join(root, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_error_histograms(store, os.path.join(plots_dir, "error_superposition_hist.png"))
    plot_multiplicative_histograms(mult_store, os.path.join(plots_dir, "multiplicative_superposition_hist.png"))

    print("\nerror-superposition gap  |e(x+y) - (e(x)+e(y))|  (target Tx+Ty):")
    print(df.round(5).to_string(index=False))
    print("\ncancellation check (error_gap should equal forecast_gap by construction):")
    print(f"  max residual across all models = {df['cancellation_residual_max'].max():.2e}")
    print("  -> confirms: on real data with target Tx+Ty, the error test equals the forecast-superposition test.")
    print("     the number is the model's error relative to a linear system with the same one-step futures.")


def plot_error_histograms(store, save_path):
    models = list(store.keys())
    palette = ["tab:red", "tab:green", "tab:purple", "tab:brown"]
    ncols = len(models)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4), squeeze=False, sharey=True)
    axes = axes[0]

    all_vals = np.concatenate([store[m]["error_gap"] for m in models])
    bins = np.linspace(0, np.percentile(all_vals, 99), 30)

    for ax, m in zip(axes, models):
        vals = store[m]["error_gap"]
        c = palette[models.index(m) % len(palette)]
        ax.hist(vals, bins=bins, color=c, alpha=0.8)
        ax.axvline(np.median(vals), color="black", linewidth=1.2, linestyle="--",
                   label=f"median={np.median(vals):.3f}")
        ax.set_title(m)
        ax.set_xlabel("error-superposition gap")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("windows")
    fig.suptitle("Error superposition on ETTm1  |  target = Tx + Ty (linear-equivalent target)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")



def plot_multiplicative_histograms(store, save_path):
    models = list(store.keys())
    palette = ["tab:red", "tab:green", "tab:purple", "tab:brown"]
    ncols = len(models)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4), squeeze=False, sharey=True)
    axes = axes[0]

    all_vals = np.concatenate([store[m]["gap"] for m in models])
    bins = np.linspace(0, np.percentile(all_vals, 99), 30)

    for ax, m in zip(axes, models):
        vals = store[m]["gap"]
        c = palette[models.index(m) % len(palette)]
        ax.hist(vals, bins=bins, color=c, alpha=0.8)
        ax.axvline(np.median(vals), color="black", linewidth=1.2, linestyle="--",
                   label=f"median={np.median(vals):.3f}")
        ax.set_title(m)
        ax.set_xlabel("multiplicative gap")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("windows")
    fig.suptitle("Multiplicative superposition on ETTm1  |  |f(x*y) - f(x)*f(y)|  (forecast space)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()