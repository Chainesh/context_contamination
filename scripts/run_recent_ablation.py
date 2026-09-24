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


def segment_positions(k, removable_region, rng):
    start_lo, start_hi = removable_region[0], removable_region[1] - k
    if start_hi < start_lo:
        raise ValueError(f"segment length {k} does not fit in removable region {removable_region}")
    seg_start = rng.integers(start_lo, start_hi + 1)
    return np.arange(seg_start, seg_start + k)


def remove_and_interpolate(context, positions):
    L = len(context)
    keep = np.ones(L, dtype=bool)
    keep[positions] = False
    out = context.copy()
    idx = np.arange(L)
    out[~keep] = np.interp(idx[~keep], idx[keep], context[keep])
    return out


def remove_batch(contexts, k, removable_region, rng):
    out = np.empty_like(contexts)
    for i in range(len(contexts)):
        positions = segment_positions(k, removable_region, rng)
        out[i] = remove_and_interpolate(contexts[i], positions)
    return out


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
    p.add_argument("--region-start", type=int, default=384)
    p.add_argument("--region-end", type=int, default=512)
    p.add_argument("--protect-recent", type=int, default=64,
                    help="never remove points in the last N positions before the forecast boundary")
    p.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    p.add_argument("--seeds", type=int, default=5)
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
    removable_region = (region[0], region[1] - args.protect_recent)
    print(f"windows={len(contexts)}  region={region}  protect_recent={args.protect_recent}  "
          f"removable={removable_region}  ks={args.ks}  seeds={args.seeds}")

    plots_dir = os.path.join(root, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for k in args.ks:
        rng_demo = np.random.default_rng(42)
        plot_removal_example(contexts[0], k, region, removable_region, rng_demo,
                              os.path.join(plots_dir, f"removal_example_k{k}.png"))

    rows = []
    baselines = {}
    for model_name in args.models:
        print(f"evaluating {model_name}")
        model = get_model(model_name, cfg, args.device, args.batch_size,
                          train_contexts=train_contexts, train_futures=train_futures)

        base_pred = predict(model, contexts, H, mean, std)
        base_mae = mae(futures, base_pred)
        baselines[model_name] = base_mae
        rows.append({"model": model_name, "k": 0, "seed": -1, "mae": base_mae})

        for k in args.ks:
            for seed in range(args.seeds):
                rng = np.random.default_rng(1000 * seed + k)
                removed = remove_batch(contexts, k, removable_region, rng)
                pred = predict(model, removed, H, mean, std)
                rows.append({"model": model_name, "k": k, "seed": seed, "mae": mae(futures, pred)})
        del model

    df = pd.DataFrame(rows)
    tables_dir = os.path.join(root, "results", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    df.to_csv(os.path.join(tables_dir, "recent_removal_ablation.csv"), index=False)

    plot_ablation(df, baselines, args.ks, removable_region, os.path.join(plots_dir, "recent_removal_ablation.png"))

    print("\nbaseline MAE (no removal):")
    for m, v in baselines.items():
        print(f"  {m}: {v:.4f}")
    print("\nmean MAE by removed count:")
    summary = df[df.k > 0].groupby(["model", "k"])["mae"].mean().unstack()
    print(summary.round(4).to_string())


def plot_removal_example(context, k, region, removable_region, rng, save_path):
    L = len(context)
    start, end = region
    positions = segment_positions(k, removable_region, rng)
    removed_context = remove_and_interpolate(context, positions)
    seg_start, seg_end = positions[0], positions[-1] + 1

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axvspan(start, removable_region[1], color="tab:orange", alpha=0.10, label="candidate removal region")
    ax.axvspan(removable_region[1], end, color="gray", alpha=0.15, label="protected recent zone")
    ax.axvspan(seg_start, seg_end, color="tab:red", alpha=0.25, label=f"removed segment (k={k})")
    ax.plot(np.arange(L), context, color="tab:blue", linewidth=1.4, label="clean context", zorder=2)
    ax.plot(np.arange(L), removed_context, color="tab:red", linewidth=1.1, linestyle="--",
            label="context after removal + interpolation", zorder=2)
    ax.axvline(L, color="black", linewidth=1, linestyle=":", label="forecast boundary")
    ax.set_xlim(start - 20, L + 5)
    ax.set_title(f"Recent-region segment removal  |  k={k}  protected last {end - removable_region[1]} points")
    ax.set_xlabel("context position")
    ax.set_ylabel("value")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


def plot_ablation(df, baselines, ks, removable_region, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {}
    palette = ["tab:blue", "tab:red", "tab:green", "tab:purple"]

    for i, model_name in enumerate(df["model"].unique()):
        color = palette[i % len(palette)]
        colors[model_name] = color
        sub = df[(df["model"] == model_name) & (df["k"] > 0)]

        ax.scatter(sub["k"], sub["mae"], color=color, alpha=0.35, s=18, zorder=2)

        agg = sub.groupby("k")["mae"].agg(["mean", "std"]).reindex(ks)
        ax.plot(ks, agg["mean"], color=color, marker="o", linewidth=1.8, label=f"{model_name} (mean)", zorder=3)
        ax.fill_between(ks, agg["mean"] - agg["std"], agg["mean"] + agg["std"], color=color, alpha=0.15, zorder=1)

        ax.axhline(baselines[model_name], color=color, linestyle="--", linewidth=1.2,
                   label=f"{model_name} (no removal)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel(f"contiguous segment length removed from {removable_region}")
    ax.set_ylabel("MAE")
    ax.set_title("Recent-context segment removal ablation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()