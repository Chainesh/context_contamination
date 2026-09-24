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


def predict(model, contexts, horizon, mean, std, batch_size):
    out = []
    for s in range(0, len(contexts), batch_size):
        c = contexts[s:s + batch_size]
        x = c * std + mean if model.scale == "raw" else c
        p = model.predict_batch(x, horizon)
        if model.scale == "raw":
            p = (p - mean) / std
        out.append(p)
    return np.concatenate(out, axis=0)


def stl_batch(contexts, period):
    from statsmodels.tsa.seasonal import STL
    T = np.empty_like(contexts); S = np.empty_like(contexts); R = np.empty_like(contexts)
    for i in range(len(contexts)):
        r = STL(contexts[i], period=period, robust=True).fit()
        T[i], S[i], R[i] = r.trend, r.seasonal, r.resid
    return T, S, R


def mae(pred, truth):
    return np.mean(np.abs(pred - truth), axis=1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--models", nargs="+", default=["ttm", "timesfm", "chronos"])
    p.add_argument("--max-windows", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--period", type=int, default=96)
    return p.parse_args()


def main():
    a = parse_args()
    root = os.path.join(os.path.dirname(__file__), "..")
    cfg = load_config(os.path.join(root, a.config))
    set_seed(cfg["experiment"]["seed"])

    series = load_univariate_csv(os.path.join(root, cfg["data"]["path"]),
                                  cfg["data"]["target_col"], cfg["data"]["time_col"])
    tr, va, te = train_val_test_split(series, cfg["data"]["train_fraction"],
                                       cfg["data"]["val_fraction"], cfg["data"]["test_fraction"])
    trn, van, ten, mean, std = normalize_with_train_stats(tr, va, te)

    L, H = cfg["data"]["context_length"], cfg["data"]["horizon"]
    ctx, fut = make_windows(ten, L, H, stride=cfg["data"]["stride"], max_windows=a.max_windows)
    trc, trf = make_windows(trn, L, H, stride=cfg["data"].get("train_stride", 48), max_windows=None)

    T, S, R = stl_batch(ctx, a.period)
    print(f"windows={len(ctx)} period={a.period}")

    rows = []
    for name in a.models:
        print(f"evaluating {name}")
        m = get_model(name, cfg, a.device, a.batch_size, train_contexts=trc, train_futures=trf)

        fT = predict(m, T, H, mean, std, a.batch_size)
        fS = predict(m, S, H, mean, std, a.batch_size)
        fR = predict(m, R, H, mean, std, a.batch_size)

        # mode A: leave one out of the SUMMED forecast
        sum_all = mae(fT + fS + fR, fut).mean()
        sum_noT = mae(fS + fR, fut).mean()
        sum_noS = mae(fT + fR, fut).mean()
        sum_noR = mae(fT + fS, fut).mean()

        # mode B: remove component from CONTEXT, then forecast directly
        ctx_all = mae(predict(m, T + S + R, H, mean, std, a.batch_size), fut).mean()
        ctx_noT = mae(predict(m, S + R, H, mean, std, a.batch_size), fut).mean()
        ctx_noS = mae(predict(m, T + R, H, mean, std, a.batch_size), fut).mean()
        ctx_noR = mae(predict(m, T + S, H, mean, std, a.batch_size), fut).mean()

        for mode, all_, nT, nS, nR in [("sum", sum_all, sum_noT, sum_noS, sum_noR),
                                        ("context", ctx_all, ctx_noT, ctx_noS, ctx_noR)]:
            rows.append({"model": name, "mode": mode, "mae_all": all_,
                          "mae_drop_trend": nT, "mae_drop_seasonal": nS, "mae_drop_residual": nR,
                          "contrib_trend": nT - all_, "contrib_seasonal": nS - all_,
                          "contrib_residual": nR - all_})
        del m

    df = pd.DataFrame(rows)
    td = os.path.join(root, "results", "tables"); os.makedirs(td, exist_ok=True)
    df.to_csv(os.path.join(td, "component_contribution.csv"), index=False)

    pd_ = os.path.join(root, "results", "plots"); os.makedirs(pd_, exist_ok=True)
    plot_contrib(df, os.path.join(pd_, "component_contribution.png"))

    print("\ncontribution = MAE(without component) - MAE(all). higher = more important")
    print(df.round(4).to_string(index=False))


def plot_contrib(df, path):
    modes = df["mode"].unique()
    fig, axes = plt.subplots(1, len(modes), figsize=(6.5 * len(modes), 4.5), squeeze=False)
    axes = axes[0]
    comps = ["trend", "seasonal", "residual"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for ax, mode in zip(axes, modes):
        sub = df[df["mode"] == mode]
        models = sub["model"].tolist()
        x = np.arange(len(models)); w = 0.8 / len(comps)
        for i, (c, col) in enumerate(zip(comps, colors)):
            ax.bar(x + i * w, sub[f"contrib_{c}"], w, label=c, color=col)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x + w); ax.set_xticklabels(models)
        ax.set_ylabel("MAE increase when removed")
        ax.set_title(f"mode = {mode}")
        ax.legend(fontsize=8)
    fig.suptitle("Component contribution (higher bar = component matters more)")
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()