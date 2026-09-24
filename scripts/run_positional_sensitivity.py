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
from metrics import mae


def remove_and_interpolate(context, positions):
    L = len(context)
    keep = np.ones(L, dtype=bool)
    keep[positions] = False
    out = context.copy()
    idx = np.arange(L)
    out[~keep] = np.interp(idx[~keep], idx[keep], context[keep])
    return out


def remove_batch_fixed(contexts, start, k):
    positions = np.arange(start, start + k)
    out = np.empty_like(contexts)
    for i in range(len(contexts)):
        out[i] = remove_and_interpolate(contexts[i], positions)
    return out


def sweep_positions(context_length, k, stride, region):
    start_lo, start_hi = region[0], region[1] - k
    if start_hi < start_lo:
        raise ValueError(f"segment length {k} does not fit in region {region}")
    return list(range(start_lo, start_hi + 1, stride))


def predict(model, contexts, horizon, mean, std):
    x = contexts * std + mean if model.scale == "raw" else contexts
    pred = model.predict_batch(x, horizon)
    if model.scale == "raw":
        pred = (pred - mean) / std
    return pred


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--models", nargs="+", default=["ttm", "timesfm"])
    p.add_argument("--max-windows", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--region-start", type=int, default=0)
    p.add_argument("--region-end", type=int, default=512)
    p.add_argument("--segment-lengths", type=int, nargs="+", default=[16])
    p.add_argument("--stride", type=int, default=32,
                    help="step between successive removal window positions; smaller = finer curve, slower run")
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

    region = (args.region_start, args.region_end)
    print(f"windows={len(contexts)}  region={region}  segment_lengths={args.segment_lengths}  stride={args.stride}")

    rows = []
    baselines = {}
    for model_name in args.models:
        print(f"evaluating {model_name}")
        model = get_model(model_name, cfg, args.device, args.batch_size,
                          train_contexts=train_contexts, train_futures=train_futures)

        base_pred = predict(model, contexts, H, mean, std)
        base_mae = mae(futures, base_pred)
        baselines[model_name] = base_mae

        for k in args.segment_lengths:
            positions = sweep_positions(L, k, args.stride, region)
            for start in positions:
                removed = remove_batch_fixed(contexts, start, k)
                pred = predict(model, removed, H, mean, std)
                m = mae(futures, pred)
                rows.append({
                    "model": model_name, "k": k, "start": start, "center": start + k / 2,
                    "mae": m, "baseline_mae": base_mae, "relative_degradation": m / base_mae,
                })
        del model

    df = pd.DataFrame(rows)
    tables_dir = os.path.join(root, "results", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    df.to_csv(os.path.join(tables_dir, "positional_sensitivity.csv"), index=False)

    plots_dir = os.path.join(root, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_positional_sensitivity(df, L, os.path.join(plots_dir, "positional_sensitivity.png"))
    plot_context_and_sensitivity(contexts[0], df, L, os.path.join(plots_dir, "positional_sensitivity_with_context.png"))

    print("\nbaseline MAE (no removal):")
    for m, v in baselines.items():
        print(f"  {m}: {v:.4f}")
    print("\nmost damaging position per model/k (highest relative degradation):")
    idx = df.groupby(["model", "k"])["relative_degradation"].idxmax()
    print(df.loc[idx, ["model", "k", "start", "relative_degradation"]].to_string(index=False))


def plot_positional_sensitivity(df, context_length, save_path):
    ks = sorted(df["k"].unique())
    models = df["model"].unique()
    palette = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    linestyles = ["-", "--", ":", "-."]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model_name in enumerate(models):
        color = palette[i % len(palette)]
        for j, k in enumerate(ks):
            sub = df[(df["model"] == model_name) & (df["k"] == k)].sort_values("center")
            ax.plot(sub["center"], sub["relative_degradation"], color=color,
                    linestyle=linestyles[j % len(linestyles)], marker="o", markersize=3,
                    label=f"{model_name}  k={k}")

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax.axvline(context_length, color="black", linewidth=1, linestyle=":", label="forecast boundary")
    ax.set_xlabel("center of removed segment (context position)")
    ax.set_ylabel("relative degradation (MAE / baseline MAE)")
    ax.set_title("Positional sensitivity: where removal hurts most")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


def plot_context_and_sensitivity(context_example, df, context_length, save_path):
    ks = sorted(df["k"].unique())
    models = df["model"].unique()
    palette = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    linestyles = ["-", "--", ":", "-."]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1.3]})

    ax1.plot(np.arange(context_length), context_example, color="black", linewidth=1.2)
    ax1.set_ylabel("value")
    ax1.set_title("Example context window (for spatial reference)")
    ax1.axvline(context_length, color="black", linewidth=1, linestyle=":")

    for i, model_name in enumerate(models):
        color = palette[i % len(palette)]
        for j, k in enumerate(ks):
            sub = df[(df["model"] == model_name) & (df["k"] == k)].sort_values("center")
            ax2.plot(sub["center"], sub["relative_degradation"], color=color,
                      linestyle=linestyles[j % len(linestyles)], marker="o", markersize=3,
                      label=f"{model_name}  k={k}")

    ax2.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax2.axvline(context_length, color="black", linewidth=1, linestyle=":", label="forecast boundary")
    ax2.set_xlabel("context position")
    ax2.set_ylabel("relative degradation")
    ax2.set_title("Positional sensitivity (aligned to the window above)")
    ax2.legend(fontsize=8, ncol=2)
    ax2.set_xlim(0, context_length + 5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()