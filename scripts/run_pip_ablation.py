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


def pip_importance_order(context, region):
    start, end = region
    seg = context[start:end]
    n = len(seg)
    kept = [0, n - 1]
    order = []

    def perp_dist(i, a, b):
        if b == a:
            return abs(seg[i] - seg[a])
        t = (i - a) / (b - a)
        line_val = seg[a] + t * (seg[b] - seg[a])
        return abs(seg[i] - line_val)

    while len(kept) < n:
        kept_sorted = sorted(kept)
        best_i, best_d = None, -1.0
        for si in range(len(kept_sorted) - 1):
            a, b = kept_sorted[si], kept_sorted[si + 1]
            for i in range(a + 1, b):
                d = perp_dist(i, a, b)
                if d > best_d:
                    best_d, best_i = d, i
        if best_i is None:
            break
        kept.append(best_i)
        order.append(best_i)
    return np.array([start + i for i in order], dtype=int)


def remove_and_interpolate(context, positions):
    L = len(context)
    keep = np.ones(L, dtype=bool)
    keep[positions] = False
    out = context.copy()
    idx = np.arange(L)
    out[~keep] = np.interp(idx[~keep], idx[keep], context[keep])
    return out


def remove_pip_batch(contexts, region, m, mode, rng):
    out = np.empty_like(contexts)
    for i in range(len(contexts)):
        order = pip_importance_order(contexts[i], region)
        if mode == "important":
            chosen = order[:m]
        elif mode == "unimportant":
            chosen = order[-m:]
        elif mode == "random":
            pool = np.arange(region[0], region[1])
            chosen = rng.choice(pool, size=min(m, len(pool)), replace=False)
        else:
            raise ValueError(mode)
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
    p.add_argument("--ms", type=int, nargs="+", default=[2, 4, 8, 16, 32])
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
    print(f"windows={len(contexts)}  region={region}  ms={args.ms}  seeds={args.seeds}")

    modes = ["important", "unimportant", "random"]
    rows = []
    baselines = {}
    for model_name in args.models:
        print(f"evaluating {model_name}")
        model = get_model(model_name, cfg, args.device, args.batch_size,
                          train_contexts=train_contexts, train_futures=train_futures)

        base_pred = predict(model, contexts, H, mean, std)
        base_mae = mae(futures, base_pred)
        baselines[model_name] = base_mae
        rows.append({"model": model_name, "mode": "none", "m": 0, "seed": -1, "mae": base_mae})

        for mode in modes:
            for m in args.ms:
                seeds = args.seeds if mode == "random" else 1
                for seed in range(seeds):
                    rng = np.random.default_rng(1000 * seed + m)
                    removed = remove_pip_batch(contexts, region, m, mode, rng)
                    pred = predict(model, removed, H, mean, std)
                    rows.append({"model": model_name, "mode": mode, "m": m, "seed": seed,
                                 "mae": mae(futures, pred)})
        del model

    df = pd.DataFrame(rows)
    tables_dir = os.path.join(root, "results", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    df.to_csv(os.path.join(tables_dir, "pip_ablation.csv"), index=False)

    plots_dir = os.path.join(root, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_pip_side_by_side(contexts[0], df, baselines, region, args.ms[len(args.ms) // 2],
                          os.path.join(plots_dir, "pip_side_by_side.png"))

    print("\nbaseline MAE (no removal):")
    for m, v in baselines.items():
        print(f"  {m}: {v:.4f}")
    print("\nmean MAE by removal mode and count:")
    summary = df[df["mode"] != "none"].groupby(["model", "mode", "m"])["mae"].mean().unstack()
    print(summary.round(4).to_string())


def plot_pip_side_by_side(context_example, df, baselines, region, m_demo, save_path):
    L = len(context_example)
    start, end = region
    order = pip_importance_order(context_example, region)
    important = order[:m_demo]
    unimportant = order[-m_demo:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.axvspan(start, end, color="tab:orange", alpha=0.08)
    ax1.plot(np.arange(L), context_example, color="tab:blue", linewidth=1.4, label="context", zorder=2)
    ax1.scatter(important, context_example[important], color="tab:red", marker="x", s=70,
                label=f"top-{m_demo} important (PIP)", zorder=4)
    ax1.scatter(unimportant, context_example[unimportant], color="tab:green", marker="o", s=35,
                label=f"bottom-{m_demo} unimportant", zorder=3)
    ax1.axvline(L, color="black", linewidth=1, linestyle=":", label="forecast boundary")
    ax1.set_xlim(start - 20, L + 5)
    ax1.set_title(f"Which points PIP calls important vs not  (m={m_demo})")
    ax1.set_xlabel("context position")
    ax1.set_ylabel("value")
    ax1.legend(fontsize=8, loc="upper left")

    palette = {"important": "tab:red", "unimportant": "tab:green", "random": "tab:gray"}
    models = df["model"].unique()
    linestyles = ["-", "--", ":", "-."]
    for j, model_name in enumerate(models):
        for mode in ["important", "unimportant", "random"]:
            sub = df[(df["model"] == model_name) & (df["mode"] == mode)]
            if sub.empty:
                continue
            agg = sub.groupby("m")["mae"].mean().sort_index()
            ax2.plot(agg.index, agg.values, color=palette[mode], linestyle=linestyles[j % len(linestyles)],
                     marker="o", markersize=4, label=f"{model_name}  {mode}")
    for j, model_name in enumerate(models):
        ax2.axhline(baselines[model_name], color="black", linestyle=linestyles[j % len(linestyles)],
                    linewidth=0.9, alpha=0.5)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("number of points removed")
    ax2.set_ylabel("MAE")
    ax2.set_title("Performance: removing important vs unimportant points")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()