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


def per_window_mae(futures, preds):
    return np.mean(np.abs(futures - preds), axis=1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--models", nargs="+", default=["ttm", "timesfm"])
    p.add_argument("--max-windows", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--top-n", type=int, default=20)
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

    N = len(contexts)
    print(f"windows={N}  top_n={args.top_n}  models={args.models}")

    mae_per_model = {}
    pred_per_model = {}
    for model_name in args.models:
        print(f"evaluating {model_name}")
        model = get_model(model_name, cfg, args.device, args.batch_size,
                          train_contexts=train_contexts, train_futures=train_futures)
        preds = predict(model, contexts, H, mean, std)
        mae_per_model[model_name] = per_window_mae(futures, preds)
        pred_per_model[model_name] = preds
        del model

    mae_df = pd.DataFrame(mae_per_model)
    mae_df.index.name = "window"

    worst_sets = {}
    for model_name in args.models:
        worst_sets[model_name] = set(mae_df[model_name].nlargest(args.top_n).index.tolist())

    tables_dir = os.path.join(root, "results", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    mae_df.to_csv(os.path.join(tables_dir, "per_window_mae.csv"))

    print(f"\ntop-{args.top_n} worst windows per model:")
    for model_name in args.models:
        idx = sorted(worst_sets[model_name])
        print(f"  {model_name}: {idx}")

    if len(args.models) >= 2:
        print("\npairwise overlap of worst-window sets:")
        for i in range(len(args.models)):
            for j in range(i + 1, len(args.models)):
                a, b = args.models[i], args.models[j]
                inter = worst_sets[a] & worst_sets[b]
                jac = len(inter) / len(worst_sets[a] | worst_sets[b])
                print(f"  {a} vs {b}: {len(inter)}/{args.top_n} shared  (Jaccard={jac:.2f})")

        common = set.intersection(*worst_sets.values())
        print(f"\nwindows in EVERY model's worst-{args.top_n}: {sorted(common)}")

    plots_dir = os.path.join(root, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    ref_model = args.models[0]
    worst_ranked = mae_df[ref_model].nlargest(args.top_n).index.tolist()
    plot_worst_windows(contexts, futures, pred_per_model, worst_ranked, args.models, L,
                       os.path.join(plots_dir, "worst_windows.png"))
    plot_worst_overlap_bar(mae_df, worst_sets, args.models, args.top_n,
                           os.path.join(plots_dir, "worst_window_overlap.png"))


def plot_worst_windows(contexts, futures, pred_per_model, worst_idx, models, context_length, save_path):
    n = len(worst_idx)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    palette = ["tab:red", "tab:green", "tab:purple", "tab:orange"]

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.4 * nrows), squeeze=False)
    ctx_x = np.arange(context_length)
    fut_x = np.arange(context_length, context_length + futures.shape[1])

    for cell, w in enumerate(worst_idx):
        ax = axes[cell // ncols][cell % ncols]
        ax.plot(ctx_x, contexts[w], color="black", linewidth=0.8)
        ax.plot(fut_x, futures[w], color="tab:blue", linewidth=1.2, label="truth")
        for m_i, model_name in enumerate(models):
            ax.plot(fut_x, pred_per_model[model_name][w], color=palette[m_i % len(palette)],
                    linewidth=1.0, linestyle="--", label=model_name)
        ax.axvline(context_length, color="gray", linewidth=0.6, linestyle=":")
        ax.set_title(f"window {w}", fontsize=8)
        ax.tick_params(labelsize=6)
        if cell == 0:
            ax.legend(fontsize=6, loc="upper left")

    for cell in range(n, nrows * ncols):
        axes[cell // ncols][cell % ncols].axis("off")

    fig.suptitle(f"Top-{n} worst windows (ranked by {models[0]})", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


def plot_worst_overlap_bar(mae_df, worst_sets, models, top_n, save_path):
    all_worst = sorted(set.union(*worst_sets.values()))
    fig, ax = plt.subplots(figsize=(min(14, 0.4 * len(all_worst) + 3), 4.5))

    width = 0.8 / len(models)
    palette = ["tab:red", "tab:green", "tab:purple", "tab:orange"]
    x = np.arange(len(all_worst))

    for m_i, model_name in enumerate(models):
        vals = [mae_df.loc[w, model_name] for w in all_worst]
        ax.bar(x + m_i * width, vals, width=width, label=model_name, color=palette[m_i % len(palette)])

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(all_worst, rotation=90, fontsize=7)
    ax.set_xlabel("window index (union of all models' worst sets)")
    ax.set_ylabel("per-window MAE")
    ax.set_title(f"Per-window MAE on the union of top-{top_n} worst windows")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()