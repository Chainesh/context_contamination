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


SHAPE_CATEGORIES = ["rising", "falling", "peak", "trough", "flat"]


def classify_shape(context, region, smooth_window=5, flat_quantile=0.33, extremum_order=3):
    start, end = region
    kernel = np.ones(smooth_window) / smooth_window
    smoothed = np.convolve(context, kernel, mode="same")
    slope = np.gradient(smoothed)
    abs_slope = np.abs(slope)
    flat_thresh = np.quantile(abs_slope[start:end], flat_quantile)

    labels = {cat: [] for cat in SHAPE_CATEGORIES}
    for i in range(start, end):
        lo = max(start, i - extremum_order)
        hi = min(end, i + extremum_order + 1)
        window = context[lo:hi]
        is_peak = context[i] == window.max() and context[i] > context[lo] and context[i] >= context[hi - 1]
        is_trough = context[i] == window.min() and context[i] < context[lo] and context[i] <= context[hi - 1]
        if abs_slope[i] <= flat_thresh:
            labels["flat"].append(i)
        elif is_peak:
            labels["peak"].append(i)
        elif is_trough:
            labels["trough"].append(i)
        elif slope[i] > 0:
            labels["rising"].append(i)
        else:
            labels["falling"].append(i)
    return {k: np.array(v, dtype=int) for k, v in labels.items()}


def remove_and_interpolate(context, positions):
    L = len(context)
    keep = np.ones(L, dtype=bool)
    keep[positions] = False
    out = context.copy()
    idx = np.arange(L)
    out[~keep] = np.interp(idx[~keep], idx[keep], context[keep])
    return out


def remove_category_batch(contexts, category, region, n_remove, rng, **shape_kwargs):
    out = np.empty_like(contexts)
    for i in range(len(contexts)):
        labels = classify_shape(contexts[i], region, **shape_kwargs)
        pool = labels[category]
        if len(pool) == 0:
            out[i] = contexts[i]
            continue
        k = min(n_remove, len(pool))
        chosen = rng.choice(pool, size=k, replace=False)
        out[i] = remove_and_interpolate(contexts[i], chosen)
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
    p.add_argument("--n-remove", type=int, default=6,
                    help="how many points of each shape category to remove per window")
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
    print(f"windows={len(contexts)}  region={region}  n_remove={args.n_remove}  seeds={args.seeds}")

    plots_dir = os.path.join(root, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    for cat in SHAPE_CATEGORIES:
        plot_shape_example(contexts[0], cat, region, args.n_remove, np.random.default_rng(0),
                           os.path.join(plots_dir, f"shape_example_{cat}.png"))

    rows = []
    baselines = {}
    for model_name in args.models:
        print(f"evaluating {model_name}")
        model = get_model(model_name, cfg, args.device, args.batch_size,
                          train_contexts=train_contexts, train_futures=train_futures)

        base_pred = predict(model, contexts, H, mean, std)
        base_mae = mae(futures, base_pred)
        baselines[model_name] = base_mae
        rows.append({"model": model_name, "category": "none", "seed": -1, "mae": base_mae})

        for cat in SHAPE_CATEGORIES:
            for seed in range(args.seeds):
                rng = np.random.default_rng(1000 * seed + hash(cat) % 997)
                removed = remove_category_batch(contexts, cat, region, args.n_remove, rng)
                pred = predict(model, removed, H, mean, std)
                rows.append({"model": model_name, "category": cat, "seed": seed, "mae": mae(futures, pred)})
        del model

    df = pd.DataFrame(rows)
    tables_dir = os.path.join(root, "results", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    df.to_csv(os.path.join(tables_dir, "shape_region_ablation.csv"), index=False)

    plot_shape_ablation(df, baselines, os.path.join(plots_dir, "shape_region_ablation.png"))

    print("\nbaseline MAE (no removal):")
    for m, v in baselines.items():
        print(f"  {m}: {v:.4f}")
    print("\nmean MAE by shape category removed:")
    summary = df[df.category != "none"].groupby(["model", "category"])["mae"].mean().unstack()
    print(summary.round(4).to_string())


def plot_shape_example(context, category, region, n_remove, rng, save_path):
    L = len(context)
    start, end = region
    labels = classify_shape(context, region)
    pool = labels[category]
    chosen = rng.choice(pool, size=min(n_remove, len(pool)), replace=False) if len(pool) else np.array([], dtype=int)
    removed_context = remove_and_interpolate(context, chosen) if len(chosen) else context.copy()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axvspan(start, end, color="tab:orange", alpha=0.08, label="candidate region")
    ax.plot(np.arange(L), context, color="tab:blue", linewidth=1.4, label="clean context", zorder=2)
    ax.plot(np.arange(L), removed_context, color="tab:red", linewidth=1.1, linestyle="--",
            label="after removal + interpolation", zorder=2)
    if len(pool):
        ax.scatter(pool, context[pool], color="tab:green", marker="o", s=20, alpha=0.5,
                   label=f"all '{category}' points", zorder=3)
    if len(chosen):
        ax.scatter(chosen, context[chosen], color="tab:red", marker="x", s=60,
                   label=f"removed ({len(chosen)})", zorder=4)
    ax.axvline(L, color="black", linewidth=1, linestyle=":", label="forecast boundary")
    ax.set_xlim(start - 20, L + 5)
    ax.set_title(f"Shape category '{category}': which points are removed")
    ax.set_xlabel("context position")
    ax.set_ylabel("value")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


def plot_shape_ablation(df, baselines, save_path):
    models = df["model"].unique()
    palette = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
    x = np.arange(len(SHAPE_CATEGORIES))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, model_name in enumerate(models):
        color = palette[i % len(palette)]
        means, stds = [], []
        for cat in SHAPE_CATEGORIES:
            sub = df[(df["model"] == model_name) & (df["category"] == cat)]["mae"]
            means.append(sub.mean())
            stds.append(sub.std())
        ax.bar(x + i * width, means, width=width, yerr=stds, capsize=3, color=color, alpha=0.8,
               label=model_name)
        ax.axhline(baselines[model_name], color=color, linestyle="--", linewidth=1.2)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(SHAPE_CATEGORIES)
    ax.set_xlabel("shape category removed")
    ax.set_ylabel("MAE  (dashed line = no-removal baseline)")
    ax.set_title("Which curve shape matters most when removed")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()