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


def linear_combination_sweep(model, contexts, horizon, mean, std, partner, a_vals, b_vals, batch_size):
    x, y = contexts, contexts[partner]
    N = len(x)

    f_x = predict(model, x, horizon, mean, std)
    f_y = predict(model, y, horizon, mean, std)

    grid = np.zeros((len(a_vals), len(b_vals)))
    for ia, a in enumerate(a_vals):
        for ib, b in enumerate(b_vals):
            combined = a * x + b * y
            preds = []
            for start in range(0, N, batch_size):
                preds.append(predict(model, combined[start:start + batch_size], horizon, mean, std))
            f_combined = np.concatenate(preds, axis=0)
            target = a * f_x + b * f_y
            gap = np.mean(np.abs(f_combined - target))
            grid[ia, ib] = gap
    return grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--models", nargs="+", default=["ttm", "timesfm", "chronos"])
    p.add_argument("--max-windows", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grid-size", type=int, default=9, help="points per axis; total forward passes ~= grid_size^2 * windows")
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
    a_vals = np.linspace(-1, 1, args.grid_size)
    b_vals = np.linspace(-1, 1, args.grid_size)

    total_passes = args.grid_size ** 2 * len(contexts)
    print(f"windows={len(contexts)}  grid={args.grid_size}x{args.grid_size}  "
          f"(~{total_passes} forecasts per model, plus 2x{len(contexts)} for f(x),f(y))")

    grids = {}
    for model_name in args.models:
        print(f"evaluating {model_name}")
        model = get_model(model_name, cfg, args.device, args.batch_size,
                          train_contexts=train_contexts, train_futures=train_futures)
        grids[model_name] = linear_combination_sweep(
            model, contexts, H, mean, std, partner, a_vals, b_vals, args.batch_size
        )
        del model

    tables_dir = os.path.join(root, "results", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    rows = []
    for m, grid in grids.items():
        for ia, a in enumerate(a_vals):
            for ib, b in enumerate(b_vals):
                rows.append({"model": m, "a": a, "b": b, "gap": grid[ia, ib]})
    pd.DataFrame(rows).to_csv(os.path.join(tables_dir, "linear_combination_grid.csv"), index=False)

    plots_dir = os.path.join(root, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_grid_heatmaps(grids, a_vals, b_vals, os.path.join(plots_dir, "linear_combination_heatmap.png"))

    print("\ngap at (a=1,b=1) [pure additive test] and (a=1,b=-1) [subtraction]:")
    ia1 = np.argmin(np.abs(a_vals - 1))
    ib1 = np.argmin(np.abs(b_vals - 1))
    ibm1 = np.argmin(np.abs(b_vals + 1))
    for m, grid in grids.items():
        print(f"  {m}: (1,1)={grid[ia1, ib1]:.4f}   (1,-1)={grid[ia1, ibm1]:.4f}   "
              f"mean_over_grid={grid.mean():.4f}")


def plot_grid_heatmaps(grids, a_vals, b_vals, save_path):
    models = list(grids.keys())
    vmax = max(g.max() for g in grids.values())
    ncols = len(models)
    fig, axes = plt.subplots(1, ncols, figsize=(4.6 * ncols, 4.6), squeeze=False)
    axes = axes[0]

    for ax, m in zip(axes, models):
        im = ax.imshow(grids[m], origin="lower", extent=[b_vals[0], b_vals[-1], a_vals[0], a_vals[-1]],
                       cmap="inferno", vmin=0, vmax=vmax, aspect="auto")
        ax.axhline(0, color="cyan", linewidth=0.5, alpha=0.5)
        ax.axvline(0, color="cyan", linewidth=0.5, alpha=0.5)
        ax.set_title(m)
        ax.set_xlabel("b")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("a")

    fig.suptitle("Linear-combination gap  |f(a\u00b7x+b\u00b7y) - (a\u00b7f(x)+b\u00b7f(y))|  over a,b in [-1,1]")
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
