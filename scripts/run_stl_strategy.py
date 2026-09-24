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


def stl_decompose_batch(contexts, period):
    from statsmodels.tsa.seasonal import STL
    T = np.empty_like(contexts)
    S = np.empty_like(contexts)
    R = np.empty_like(contexts)
    for i in range(len(contexts)):
        res = STL(contexts[i], period=period, robust=True).fit()
        T[i], S[i], R[i] = res.trend, res.seasonal, res.resid
    return T, S, R


def compare_strategies(model, contexts, futures, horizon, mean, std, period, batch_size):
    T, S, R = stl_decompose_batch(contexts, period)

    def batched(arr):
        out = []
        for start in range(0, len(arr), batch_size):
            out.append(predict(model, arr[start:start + batch_size], horizon, mean, std))
        return np.concatenate(out, axis=0)

    f_direct = batched(contexts)
    f_decomposed = batched(T) + batched(S) + batched(R)

    mae_direct = np.mean(np.abs(f_direct - futures), axis=1)
    mae_decomposed = np.mean(np.abs(f_decomposed - futures), axis=1)

    return {
        "f_direct": f_direct, "f_decomposed": f_decomposed,
        "trend": T, "seasonal": S, "residual": R,
        "mae_direct": mae_direct, "mae_decomposed": mae_decomposed,
        "diff": mae_decomposed - mae_direct,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--models", nargs="+", default=["ttm", "timesfm", "chronos"])
    p.add_argument("--max-windows", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--period", type=int, default=96)
    p.add_argument("--example-window", type=int, default=0)
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

    print(f"windows={len(contexts)}  period={args.period}  models={args.models}")

    rows = []
    store = {}
    for model_name in args.models:
        print(f"evaluating {model_name}")
        model = get_model(model_name, cfg, args.device, args.batch_size,
                          train_contexts=train_contexts, train_futures=train_futures)
        r = compare_strategies(model, contexts, futures, H, mean, std, args.period, args.batch_size)
        store[model_name] = r

        win_rate = float(np.mean(r["mae_decomposed"] < r["mae_direct"]))
        rows.append({
            "model": model_name,
            "mae_direct": float(r["mae_direct"].mean()),
            "mae_decomposed": float(r["mae_decomposed"].mean()),
            "mae_direct_median": float(np.median(r["mae_direct"])),
            "mae_decomposed_median": float(np.median(r["mae_decomposed"])),
            "decomposed_win_rate": win_rate,
            "mean_diff": float(r["diff"].mean()),
            "winner": "decomposed" if r["mae_decomposed"].mean() < r["mae_direct"].mean() else "direct",
        })
        del model

    df = pd.DataFrame(rows)
    tables_dir = os.path.join(root, "results", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    df.to_csv(os.path.join(tables_dir, "stl_strategy_comparison.csv"), index=False)

    per_window = []
    for m, r in store.items():
        for w in range(len(r["mae_direct"])):
            per_window.append({"model": m, "window": w,
                               "mae_direct": float(r["mae_direct"][w]),
                               "mae_decomposed": float(r["mae_decomposed"][w]),
                               "diff": float(r["diff"][w])})
    pd.DataFrame(per_window).to_csv(os.path.join(tables_dir, "stl_strategy_per_window.csv"), index=False)

    plots_dir = os.path.join(root, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_example_with_truth(store, contexts, futures, L, H, args.example_window,
                            os.path.join(plots_dir, "stl_strategy_example.png"))
    plot_mae_comparison(df, os.path.join(plots_dir, "stl_strategy_mae.png"))
    plot_diff_histograms(store, os.path.join(plots_dir, "stl_strategy_diff_hist.png"))

    print("\ndirect forecast vs STL decompose-then-sum (lower MAE wins):")
    print(df.round(4).to_string(index=False))
    print("\nwin rate = fraction of windows where decomposed beat direct")


def plot_example_with_truth(store, contexts, futures, L, H, w, save_path):
    models = list(store.keys())
    ref = store[models[0]]
    ctx_x = np.arange(L)
    fut_x = np.arange(L, L + H)
    palette = ["tab:red", "tab:green", "tab:purple", "tab:brown"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                    gridspec_kw={"height_ratios": [1, 1.2]})

    ax1.plot(ctx_x, contexts[w], color="black", linewidth=1.0, label="context")
    ax1.plot(ctx_x, ref["trend"][w], color="tab:blue", linewidth=1.3, label="trend")
    ax1.plot(ctx_x, ref["seasonal"][w], color="tab:orange", linewidth=0.9, label="seasonal")
    ax1.plot(ctx_x, ref["residual"][w], color="tab:green", linewidth=0.7, label="residual")
    ax1.axvline(L, color="black", linewidth=1, linestyle=":")
    ax1.set_title(f"STL decomposition of window {w}")
    ax1.set_ylabel("value")
    ax1.legend(fontsize=8, ncol=2, loc="upper left")

    ax2.plot(fut_x, futures[w], color="black", linewidth=2.2, label="GROUND TRUTH", zorder=5)
    for i, m in enumerate(models):
        c = palette[i % len(palette)]
        ax2.plot(fut_x, store[m]["f_direct"][w], color=c, linewidth=1.5,
                 label=f"{m}: direct f(whole)")
        ax2.plot(fut_x, store[m]["f_decomposed"][w], color=c, linewidth=1.3, linestyle="--",
                 label=f"{m}: decomposed sum")
    ax2.set_title("Which is closer to ground truth: direct (solid) or decomposed (dashed)?")
    ax2.set_xlabel("forecast step")
    ax2.set_ylabel("value")
    ax2.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


def plot_mae_comparison(df, save_path):
    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(1.8 * len(models) + 4, 4.5))
    ax.bar(x - width / 2, df["mae_direct"], width, label="direct f(whole)", color="tab:blue")
    ax.bar(x + width / 2, df["mae_decomposed"], width, label="STL decomposed sum", color="tab:orange")
    for i, r in df.iterrows():
        ax.text(i, max(r["mae_direct"], r["mae_decomposed"]) * 1.02,
                f"win rate {r['decomposed_win_rate']:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("MAE vs ground truth")
    ax.set_title("Direct forecast vs STL decompose-then-forecast (lower is better)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


def plot_diff_histograms(store, save_path):
    models = list(store.keys())
    palette = ["tab:red", "tab:green", "tab:purple", "tab:brown"]
    all_vals = np.concatenate([store[m]["diff"] for m in models])
    lim = np.percentile(np.abs(all_vals), 99)
    bins = np.linspace(-lim, lim, 35)

    ncols = len(models)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4), squeeze=False, sharey=True)
    axes = axes[0]
    for ax, m in zip(axes, models):
        vals = store[m]["diff"]
        c = palette[models.index(m) % len(palette)]
        ax.hist(vals, bins=bins, color=c, alpha=0.8)
        ax.axvline(0, color="black", linewidth=1.4)
        ax.axvline(np.median(vals), color="blue", linewidth=1.2, linestyle="--",
                   label=f"median={np.median(vals):+.4f}")
        ax.set_title(m)
        ax.set_xlabel("MAE(decomposed) - MAE(direct)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("windows")
    fig.suptitle("Negative = decomposition helped, Positive = direct was better")
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()