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


def predict(model, contexts, horizon, mean, std):
    x = contexts * std + mean if model.scale == "raw" else contexts
    pred = model.predict_batch(x, horizon)
    if model.scale == "raw":
        pred = (pred - mean) / std
    return pred


def moving_average(contexts, window):
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(contexts, ((0, 0), (pad, pad)), mode="edge")
    trend = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="valid"), 1, padded)
    return trend[:, :contexts.shape[1]]


def additive_decompose_3(context, period):
    from statsmodels.tsa.seasonal import STL
    res = STL(context, period=period, robust=True).fit()
    return res.trend, res.seasonal, res.resid


def decompose3_batch(contexts, period):
    T = np.empty_like(contexts)
    S = np.empty_like(contexts)
    R = np.empty_like(contexts)
    for i in range(len(contexts)):
        t, s, r = additive_decompose_3(contexts[i], period)
        T[i], S[i], R[i] = t, s, r
    return T, S, R


def seasonal_decomposition_test(model, contexts, horizon, mean, std, period):
    T, S, R = decompose3_batch(contexts, period)
    f_whole = predict(model, contexts, horizon, mean, std)
    f_t = predict(model, T, horizon, mean, std)
    f_s = predict(model, S, horizon, mean, std)
    f_r = predict(model, R, horizon, mean, std)
    f_sum = f_t + f_s + f_r
    gap = np.mean(np.abs(f_whole - f_sum), axis=1)
    scale = np.mean(np.abs(f_whole), axis=1) + 1e-8
    return {
        "f_whole": f_whole, "f_sum": f_sum,
        "trend": T, "seasonal": S, "residual": R,
        "gap": gap, "relative_gap": gap / scale,
    }


def decomposition_test(model, contexts, horizon, mean, std, ma_window):
    trend = moving_average(contexts, ma_window)
    residual = contexts - trend

    f_whole = predict(model, contexts, horizon, mean, std)
    f_trend = predict(model, trend, horizon, mean, std)
    f_resid = predict(model, residual, horizon, mean, std)
    f_sum = f_trend + f_resid

    gap = np.mean(np.abs(f_whole - f_sum), axis=1)
    scale = np.mean(np.abs(f_whole), axis=1) + 1e-8
    return {
        "f_whole": f_whole, "f_sum": f_sum, "f_trend": f_trend, "f_resid": f_resid,
        "trend": trend, "residual": residual,
        "gap": gap, "relative_gap": gap / scale,
    }


def offset_test(model, contexts, horizon, mean, std, shift):
    shifted = contexts + shift
    f_base = predict(model, contexts, horizon, mean, std)
    f_shifted = predict(model, shifted, horizon, mean, std)

    gap = np.mean(np.abs(f_shifted - (f_base + shift)), axis=1)
    scale = np.mean(np.abs(f_base), axis=1) + 1e-8
    return {
        "f_base": f_base, "f_shifted": f_shifted, "shift": shift,
        "gap": gap, "relative_gap": gap / scale,
    }


def scale_test(model, contexts, horizon, mean, std, factor):
    scaled = contexts * factor
    f_base = predict(model, contexts, horizon, mean, std)
    f_scaled = predict(model, scaled, horizon, mean, std)

    gap = np.mean(np.abs(f_scaled - factor * f_base), axis=1)
    denom = np.mean(np.abs(factor * f_base), axis=1) + 1e-8
    return {
        "f_base": f_base, "f_scaled": f_scaled, "factor": factor,
        "gap": gap, "relative_gap": gap / denom,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--models", nargs="+", default=["ttm", "timesfm"])
    p.add_argument("--max-windows", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--ma-window", type=int, default=25, help="moving-average window for trend/residual split")
    p.add_argument("--shift", type=float, default=1.0, help="constant added in the offset test (normalized units)")
    p.add_argument("--factor", type=float, default=2.0, help="multiplier for the scale (RevIN std) test")
    p.add_argument("--period", type=int, default=96, help="seasonal period for 3-part decomposition (96 = daily for ETTm1)")
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

    print(f"windows={len(contexts)}  ma_window={args.ma_window}  shift={args.shift}  models={args.models}")

    rows = []
    decomp_store = {}
    offset_store = {}
    seasonal_store = {}
    scale_store = {}
    for model_name in args.models:
        print(f"evaluating {model_name}")
        model = get_model(model_name, cfg, args.device, args.batch_size,
                          train_contexts=train_contexts, train_futures=train_futures)

        d = decomposition_test(model, contexts, H, mean, std, args.ma_window)
        o = offset_test(model, contexts, H, mean, std, args.shift)
        sd = seasonal_decomposition_test(model, contexts, H, mean, std, args.period)
        sc = scale_test(model, contexts, H, mean, std, args.factor)
        decomp_store[model_name] = d
        offset_store[model_name] = o
        seasonal_store[model_name] = sd
        scale_store[model_name] = sc

        rows.append({
            "model": model_name,
            "decomposition_gap_mean": float(d["gap"].mean()),
            "decomposition_relgap_mean": float(d["relative_gap"].mean()),
            "offset_shift_gap_mean": float(o["gap"].mean()),
            "offset_shift_relgap_mean": float(o["relative_gap"].mean()),
            "scale_gap_mean": float(sc["gap"].mean()),
            "scale_relgap_mean": float(sc["relative_gap"].mean()),
            "revin_like": bool(o["relative_gap"].mean() < 0.02 and sc["relative_gap"].mean() < 0.02),
            "seasonal_gap_mean": float(sd["gap"].mean()),
            "seasonal_gap_median": float(np.median(sd["gap"])),
            "seasonal_relgap_mean": float(sd["relative_gap"].mean()),
            "seasonal_relgap_median": float(np.median(sd["relative_gap"])),
        })
        del model

    df = pd.DataFrame(rows)
    tables_dir = os.path.join(root, "results", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    df.to_csv(os.path.join(tables_dir, "linearity_test.csv"), index=False)

    plots_dir = os.path.join(root, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_summary(df, os.path.join(plots_dir, "linearity_summary.png"))
    plot_decomposition_example(decomp_store, contexts[0], L, H,
                               os.path.join(plots_dir, "linearity_decomposition_example.png"))
    plot_offset_example(offset_store, contexts[0], L, H,
                        os.path.join(plots_dir, "linearity_offset_example.png"))
    plot_seasonal_histograms(seasonal_store, "gap",
                             os.path.join(plots_dir, "linearity_seasonal_hist_raw.png"))
    plot_seasonal_histograms(seasonal_store, "relative_gap",
                             os.path.join(plots_dir, "linearity_seasonal_hist_relative.png"))
    plot_seasonal_example(seasonal_store, contexts[0], L, H, args.period,
                          os.path.join(plots_dir, "linearity_seasonal_example.png"))
    plot_revin_check(offset_store, scale_store, df,
                     os.path.join(plots_dir, "revin_check.png"))

    gap_rows = []
    for m, sd in seasonal_store.items():
        for w in range(len(sd["gap"])):
            gap_rows.append({"model": m, "window": w,
                             "gap": float(sd["gap"][w]), "relative_gap": float(sd["relative_gap"][w])})
    pd.DataFrame(gap_rows).to_csv(os.path.join(tables_dir, "linearity_seasonal_per_window.csv"), index=False)

    print("\nlinearity gaps (0 = perfectly linear; larger = more nonlinear):")
    print(df.round(4).to_string(index=False))
    print("\nRevIN check (reversible instance normalization):")
    print("  a model with RevIN drives BOTH the shift gap and scale gap to ~0")
    for _, r in df.iterrows():
        verdict = "RevIN-like" if r["revin_like"] else "not RevIN-like"
        print(f"  {r['model']}: shift_relgap={r['offset_shift_relgap_mean']:.4f}  "
              f"scale_relgap={r['scale_relgap_mean']:.4f}  ->  {verdict}")


def plot_summary(df, save_path):
    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(1.6 * len(models) + 3, 4.5))
    ax.bar(x - width / 2, df["decomposition_relgap_mean"], width, label="decomposition  |f(a)+f(b) - f(a+b)|")
    ax.bar(x + width / 2, df["offset_shift_relgap_mean"], width, label="offset  |f(x+c) - (f(x)+c)|")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("relative gap (mean over windows)")
    ax.set_title("Linearity violation by model (0 = linear)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


def plot_decomposition_example(store, context_example, L, H, save_path):
    models = list(store.keys())
    ref = store[models[0]]
    fut_x = np.arange(L, L + H)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7),
                                    gridspec_kw={"height_ratios": [1, 1.1]})

    ax1.plot(np.arange(L), context_example, color="black", linewidth=1.0, label="whole context")
    ax1.plot(np.arange(L), ref["trend"][0], color="tab:blue", linewidth=1.2, label="trend (a)")
    ax1.plot(np.arange(L), ref["residual"][0], color="tab:orange", linewidth=0.8, label="residual (b)")
    ax1.axvline(L, color="black", linewidth=1, linestyle=":")
    ax1.set_title("Context split into trend + residual (they sum to the whole)")
    ax1.set_ylabel("value")
    ax1.legend(fontsize=8, loc="upper left")

    palette = ["tab:red", "tab:green", "tab:purple", "tab:brown"]
    for i, m in enumerate(models):
        c = palette[i % len(palette)]
        ax2.plot(fut_x, store[m]["f_whole"][0], color=c, linewidth=1.6, label=f"{m}: f(a+b)")
        ax2.plot(fut_x, store[m]["f_sum"][0], color=c, linewidth=1.2, linestyle="--", label=f"{m}: f(a)+f(b)")
    ax2.set_title("Forecast of the whole vs sum of the two forecasts")
    ax2.set_xlabel("forecast step")
    ax2.set_ylabel("value")
    ax2.legend(fontsize=8, ncol=len(models))
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


def plot_offset_example(store, context_example, L, H, save_path):
    models = list(store.keys())
    shift = store[models[0]]["shift"]
    fut_x = np.arange(L, L + H)

    fig, ax = plt.subplots(figsize=(11, 5))
    palette = ["tab:red", "tab:green", "tab:purple", "tab:brown"]
    for i, m in enumerate(models):
        c = palette[i % len(palette)]
        ax.plot(fut_x, store[m]["f_shifted"][0], color=c, linewidth=1.6, label=f"{m}: f(x+c)")
        ax.plot(fut_x, store[m]["f_base"][0] + shift, color=c, linewidth=1.2, linestyle="--",
                label=f"{m}: f(x)+c")
    ax.set_title(f"Offset test: forecast of shifted input vs shifted forecast  (c={shift:g})")
    ax.set_xlabel("forecast step")
    ax.set_ylabel("value")
    ax.legend(fontsize=8, ncol=len(models))
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")



def plot_seasonal_histograms(store, key, save_path):
    models = list(store.keys())
    palette = ["tab:red", "tab:green", "tab:purple", "tab:brown"]
    ncols = len(models)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4), squeeze=False, sharey=True)
    axes = axes[0]

    all_vals = np.concatenate([store[m][key] for m in models])
    bins = np.linspace(0, np.percentile(all_vals, 99), 30)

    label = "raw gap (MAE units)" if key == "gap" else "relative gap"
    for ax, m in zip(axes, models):
        vals = store[m][key]
        c = palette[models.index(m) % len(palette)]
        ax.hist(vals, bins=bins, color=c, alpha=0.8)
        ax.axvline(np.median(vals), color="black", linewidth=1.2, linestyle="--",
                   label=f"median={np.median(vals):.3f}")
        ax.set_title(m)
        ax.set_xlabel(label)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("windows")
    fig.suptitle(f"Superposition gap distribution  |  f(trend+seasonal+residual) vs sum  ({label})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


def plot_seasonal_example(store, context_example, L, H, period, save_path):
    models = list(store.keys())
    ref = store[models[0]]
    fut_x = np.arange(L, L + H)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5),
                                    gridspec_kw={"height_ratios": [1.2, 1]})

    ax1.plot(np.arange(L), context_example, color="black", linewidth=1.0, label="whole context")
    ax1.plot(np.arange(L), ref["trend"][0], color="tab:blue", linewidth=1.3, label="trend")
    ax1.plot(np.arange(L), ref["seasonal"][0], color="tab:orange", linewidth=0.9, label="seasonal")
    ax1.plot(np.arange(L), ref["residual"][0], color="tab:green", linewidth=0.7, label="residual")
    ax1.axvline(L, color="black", linewidth=1, linestyle=":")
    ax1.set_title(f"Context split into trend + seasonal + residual  (period={period})")
    ax1.set_ylabel("value")
    ax1.legend(fontsize=8, loc="upper left", ncol=2)

    palette = ["tab:red", "tab:green", "tab:purple", "tab:brown"]
    for i, m in enumerate(models):
        c = palette[i % len(palette)]
        ax2.plot(fut_x, store[m]["f_whole"][0], color=c, linewidth=1.6, label=f"{m}: f(whole)")
        ax2.plot(fut_x, store[m]["f_sum"][0], color=c, linewidth=1.2, linestyle="--", label=f"{m}: sum of parts")
    ax2.set_title("Forecast of the whole vs sum of the three component forecasts")
    ax2.set_xlabel("forecast step")
    ax2.set_ylabel("value")
    ax2.legend(fontsize=7, ncol=len(models))
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")

def plot_revin_check(offset_store, scale_store, df, save_path):
    models = list(offset_store.keys())
    palette = ["tab:red", "tab:green", "tab:purple", "tab:brown"]
    ncols = len(models)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.2), squeeze=False, sharey=True)
    axes = axes[0]

    shift_all = np.concatenate([offset_store[m]["relative_gap"] for m in models])
    scale_all = np.concatenate([scale_store[m]["relative_gap"] for m in models])
    hi = np.percentile(np.concatenate([shift_all, scale_all]), 99) + 1e-6
    bins = np.linspace(0, hi, 30)

    for ax, m in zip(axes, models):
        c = palette[models.index(m) % len(palette)]
        ax.hist(offset_store[m]["relative_gap"], bins=bins, color=c, alpha=0.55,
                label="shift gap (mean)")
        ax.hist(scale_store[m]["relative_gap"], bins=bins, color="black", alpha=0.35,
                label="scale gap (std)")
        verdict = "RevIN-like" if df[df.model == m]["revin_like"].iloc[0] else "not RevIN-like"
        ax.set_title(f"{m}\n{verdict}")
        ax.set_xlabel("relative gap")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("windows")
    fig.suptitle("RevIN check: reversible instance normalization drives both gaps to 0")
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()