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


def pip_indices(series, m):
    n = len(series)
    selected = [0, n - 1]
    while len(selected) < m:
        sel = sorted(selected)
        best_d, best_j = -1.0, None
        for i in range(len(sel) - 1):
            a, b = sel[i], sel[i + 1]
            if b - a < 2:
                continue
            va, vb = series[a], series[b]
            for j in range(a + 1, b):
                t = (j - a) / (b - a)
                line = va + t * (vb - va)
                d = abs(series[j] - line)
                if d > best_d:
                    best_d, best_j = d, j
        if best_j is None:
            break
        selected.append(best_j)
    return np.array(sorted(selected))


def flatten_prefix_batch(contexts, cutoffs):
    out = contexts.copy()
    for i in range(len(contexts)):
        c = cutoffs[i]
        out[i, :c] = contexts[i, c]
    return out


def predict(model, contexts, horizon, mean, std):
    x = contexts * std + mean if model.scale == "raw" else contexts
    pred = model.predict_batch(x, horizon)
    if model.scale == "raw":
        pred = (pred - mean) / std
    return pred


def per_window_mae(futures, preds):
    return np.abs(futures - preds).mean(axis=1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--models", nargs="+", default=["ttm", "timesfm"])
    p.add_argument("--max-windows", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--m", type=int, default=8, help="number of PIP points per window")
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

    M = args.m
    pip_all = np.stack([pip_indices(c, M) for c in contexts])
    mean_positions = pip_all.mean(axis=0)
    order = np.argsort(mean_positions)
    mean_positions_sorted = mean_positions[order]
    print(f"windows={len(contexts)}  M={M}  mean PIP positions={mean_positions.round(1)}")

    plots_dir = os.path.join(root, "results", "plots")
    tables_dir = os.path.join(root, "results", "tables")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    rows = []
    baselines = {}
    for model_name in args.models:
        print(f"evaluating {model_name}")
        model = get_model(model_name, cfg, args.device, args.batch_size,
                          train_contexts=train_contexts, train_futures=train_futures)

        base_pw = per_window_mae(futures, predict(model, contexts, H, mean, std))
        base_mae = float(base_pw.mean())
        baselines[model_name] = base_mae

        reldeg = np.zeros((len(contexts), M))
        for col, rank in enumerate(order):
            cutoffs = pip_all[:, rank]
            flattened = flatten_prefix_batch(contexts, cutoffs)
            pw = per_window_mae(futures, predict(model, flattened, H, mean, std))
            reldeg[:, col] = pw / base_pw
            rows.append({
                "model": model_name, "rank_from_right": M - 1 - rank,
                "mean_cutoff": float(cutoffs.mean()), "mae": float(pw.mean()),
                "baseline_mae": base_mae, "relative_degradation": float(pw.mean()) / base_mae,
            })

        plot_pip_heatmap(reldeg, mean_positions_sorted, model_name,
                         os.path.join(plots_dir, f"pip_prefix_heatmap_{model_name}.png"))
        del model

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(tables_dir, "pip_prefix_flatten.csv"), index=False)

    print("\nbaseline MAE (no flattening):")
    for m, v in baselines.items():
        print(f"  {m}: {v:.4f}")
    print("\npooled MAE by cutoff (rank from right, mean position):")
    print(df[["model", "rank_from_right", "mean_cutoff", "mae", "relative_degradation"]].round(4).to_string(index=False))


def plot_pip_heatmap(reldeg, mean_positions_sorted, model_name, save_path):
    N, M = reldeg.shape
    win_order = np.argsort(reldeg[:, -1])[::-1]
    reldeg_sorted = reldeg[win_order]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [2.2, 1]})

    im = ax1.imshow(reldeg_sorted, aspect="auto", cmap="RdBu_r", vmin=0.7, vmax=1.3,
                    extent=[0, M, 0, N], origin="lower")
    ax1.set_xticks(np.arange(M) + 0.5)
    ax1.set_xticklabels([str(int(p)) for p in mean_positions_sorted], fontsize=8)
    ax1.set_xlabel("cutoff position (mean over windows; left=baseline, right=fully flattened)")
    ax1.set_ylabel("test window (sorted by sensitivity)")
    ax1.set_title(f"Per-window relative degradation under prefix flattening ({model_name})")
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label("MAE(cutoff) / MAE(baseline)")

    med = np.median(reldeg, axis=0)
    q1 = np.percentile(reldeg, 25, axis=0)
    q3 = np.percentile(reldeg, 75, axis=0)
    ax2.plot(mean_positions_sorted, med, color="tab:red", marker="o", label="median")
    ax2.fill_between(mean_positions_sorted, q1, q3, color="tab:red", alpha=0.2, label="IQR (25-75%)")
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax2.set_xlabel("cutoff position")
    ax2.set_ylabel("relative degradation")
    ax2.set_title("Median and spread across windows")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()