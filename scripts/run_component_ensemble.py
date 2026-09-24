import argparse
import itertools
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

COMPONENTS = ["T", "S", "R"]


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


def load_dataset(root, cfg_path, period_override, max_windows):
    cfg = load_config(os.path.join(root, cfg_path))
    series = load_univariate_csv(os.path.join(root, cfg["data"]["path"]),
                                  cfg["data"]["target_col"], cfg["data"]["time_col"])
    tr, va, te = train_val_test_split(series, cfg["data"]["train_fraction"],
                                       cfg["data"]["val_fraction"], cfg["data"]["test_fraction"])
    trn, van, ten, mean, std = normalize_with_train_stats(tr, va, te)
    L, H = cfg["data"]["context_length"], cfg["data"]["horizon"]
    ctx, fut = make_windows(ten, L, H, stride=cfg["data"]["stride"], max_windows=max_windows)
    trc, trf = make_windows(trn, L, H, stride=cfg["data"].get("train_stride", 48), max_windows=None)
    period = period_override if period_override else cfg["data"].get("period", 96)
    return dict(cfg=cfg, name=cfg["data"].get("name", os.path.basename(cfg_path)),
                ctx=ctx, fut=fut, trc=trc, trf=trf, mean=mean, std=std, H=H, period=period)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=["configs/default.yaml", "configs/us_births.yaml"],
                    help="one or more dataset configs; selection is done JOINTLY across all of them")
    p.add_argument("--models", nargs="+", default=["ttm", "timesfm", "chronos"])
    p.add_argument("--max-windows", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--period", type=int, default=None, help="override each config's period")
    p.add_argument("--select-frac", type=float, default=0.5)
    p.add_argument("--plots-dir", default="results/plots/ensemble")
    return p.parse_args()


def main():
    a = parse_args()
    root = os.path.join(os.path.dirname(__file__), "..")

    datasets = []
    for cp in a.configs:
        d = load_dataset(root, cp, a.period, a.max_windows)
        set_seed(d["cfg"]["experiment"]["seed"])
        d["T"], d["S"], d["R"] = stl_batch(d["ctx"], d["period"])
        N = len(d["ctx"]); n_sel = int(N * a.select_frac)
        d["sel"] = np.arange(n_sel); d["ev"] = np.arange(n_sel, N)
        print(f"{d['name']}: windows={N} select={n_sel} eval={N-n_sel} period={d['period']}")
        datasets.append(d)

    # forecast every component with every model, per dataset
    for d in datasets:
        d["fc"] = {}
    for name in a.models:
        print(f"evaluating {name}")
        for d in datasets:
            m = get_model(name, d["cfg"], a.device, a.batch_size,
                          train_contexts=d["trc"], train_futures=d["trf"])
            d["fc"][name] = {
                "T": predict(m, d["T"], d["H"], d["mean"], d["std"], a.batch_size),
                "S": predict(m, d["S"], d["H"], d["mean"], d["std"], a.batch_size),
                "R": predict(m, d["R"], d["H"], d["mean"], d["std"], a.batch_size),
                "direct": predict(m, d["ctx"], d["H"], d["mean"], d["std"], a.batch_size),
            }
            del m

    def combo_mae(d, combo, idx):
        cT, cS, cR = combo
        p = d["fc"][cT]["T"][idx] + d["fc"][cS]["S"][idx] + d["fc"][cR]["R"][idx]
        return mae(p, d["fut"][idx]).mean()

    # JOINT selection: normalise each dataset by its own best single-model direct MAE,
    # so neither dataset's scale dominates the combined objective
    ref = {}
    for d in datasets:
        ref[d["name"]] = min(mae(d["fc"][n]["direct"][d["sel"]], d["fut"][d["sel"]]).mean()
                              for n in a.models)

    combos = list(itertools.product(a.models, repeat=3))
    joint = {}
    per_ds_sel = {d["name"]: {} for d in datasets}
    for combo in combos:
        tot = 0.0
        for d in datasets:
            v = combo_mae(d, combo, d["sel"])
            per_ds_sel[d["name"]][combo] = v
            tot += v / ref[d["name"]]
        joint[combo] = tot / len(datasets)
    best_joint = min(joint, key=joint.get)
    best_per_ds = {d["name"]: min(per_ds_sel[d["name"]], key=per_ds_sel[d["name"]].get)
                    for d in datasets}

    rows = []
    for d in datasets:
        ev, fut = d["ev"], d["fut"]
        for n in a.models:
            rows.append({"dataset": d["name"], "method": f"{n} (direct)",
                          "mae_eval": mae(d["fc"][n]["direct"][ev], fut[ev]).mean()})
            rows.append({"dataset": d["name"], "method": f"{n} (decomposed)",
                          "mae_eval": combo_mae(d, (n, n, n), ev)})
        avg = np.mean([d["fc"][n]["direct"][ev] for n in a.models], axis=0)
        rows.append({"dataset": d["name"], "method": "mean of direct",
                      "mae_eval": mae(avg, fut[ev]).mean()})
        rows.append({"dataset": d["name"],
                      "method": f"ENSEMBLE-joint T={best_joint[0]} S={best_joint[1]} R={best_joint[2]}",
                      "mae_eval": combo_mae(d, best_joint, ev)})
        bp = best_per_ds[d["name"]]
        rows.append({"dataset": d["name"],
                      "method": f"ENSEMBLE-per-dataset T={bp[0]} S={bp[1]} R={bp[2]}",
                      "mae_eval": combo_mae(d, bp, ev)})

    df = pd.DataFrame(rows)
    td = os.path.join(root, "results", "tables"); os.makedirs(td, exist_ok=True)
    df.to_csv(os.path.join(td, "component_ensemble_multi.csv"), index=False)
    pd.DataFrame([{"trend": k[0], "seasonal": k[1], "residual": k[2], "joint_score": v}
                   for k, v in joint.items()]).sort_values("joint_score").to_csv(
        os.path.join(td, "component_ensemble_joint_combos.csv"), index=False)

    pdir = os.path.join(root, a.plots_dir); os.makedirs(pdir, exist_ok=True)
    plot_multi(df, datasets, os.path.join(pdir, "ensemble_multi_dataset.png"))
    plot_joint_heat(joint, a.models, os.path.join(pdir, "ensemble_joint_combos.png"))
    plot_improvement(df, datasets, a.models, os.path.join(pdir, "ensemble_improvement.png"))
    plot_component_table(joint, per_ds_sel, datasets, a.models,
                         os.path.join(pdir, "ensemble_best_model_per_component.png"))

    print(f"\njoint best (selected across all datasets): T={best_joint[0]} S={best_joint[1]} R={best_joint[2]}")
    for d in datasets:
        bp = best_per_ds[d["name"]]
        print(f"  {d['name']} own best: T={bp[0]} S={bp[1]} R={bp[2]}")
    print("\neval-half MAE:")
    for d in datasets:
        sub = df[df.dataset == d["name"]].sort_values("mae_eval")
        print(f"\n[{d['name']}]")
        print(sub[["method", "mae_eval"]].round(4).to_string(index=False))
    print(f"\nplots -> {a.plots_dir}")


def plot_multi(df, datasets, path):
    names = [d["name"] for d in datasets]
    fig, axes = plt.subplots(1, len(names), figsize=(7.5 * len(names), 5), squeeze=False)
    axes = axes[0]
    for ax, nm in zip(axes, names):
        sub = df[df.dataset == nm].sort_values("mae_eval")
        colors = ["tab:red" if m.startswith("ENSEMBLE-joint")
                  else "tab:orange" if m.startswith("ENSEMBLE-per")
                  else "tab:gray" if m.startswith("mean of") else "tab:blue"
                  for m in sub["method"]]
        ax.barh(sub["method"][::-1], sub["mae_eval"][::-1], color=colors[::-1])
        ax.set_xlabel("MAE (held-out eval windows)")
        ax.set_title(nm)
    fig.suptitle("Per-component ensemble vs baselines (red = jointly selected)")
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight")


def plot_joint_heat(joint, models, path):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), squeeze=False)
    axes = axes[0]
    vals = np.array(list(joint.values()))
    vmin, vmax = vals.min(), vals.max()
    for ax, mR in zip(axes, models):
        grid = np.zeros((n, n))
        for i, mT in enumerate(models):
            for j, mS in enumerate(models):
                grid[i, j] = joint[(mT, mS, mR)]
        im = ax.imshow(grid, cmap="viridis_r", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(n)); ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(models, fontsize=8)
        ax.set_xlabel("seasonal"); ax.set_title(f"residual = {mR}")
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if grid[i, j] > (vmin + vmax) / 2 else "black")
    axes[0].set_ylabel("trend")
    fig.colorbar(im, ax=axes[-1], fraction=0.046)
    fig.suptitle("Joint selection score across all datasets (lower = better)")
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight")

def plot_improvement(df, datasets, models, path):
    names = [d["name"] for d in datasets]
    fig, axes = plt.subplots(1, len(names), figsize=(6.5 * len(names), 4.2), squeeze=False)
    axes = axes[0]
    for ax, nm in zip(axes, names):
        sub = df[df.dataset == nm]
        singles = sub[sub.method.isin([f"{m} (direct)" for m in models])]
        best_row = singles.loc[singles.mae_eval.idxmin()]
        best = best_row.mae_eval
        best_name = best_row.method.replace(" (direct)", "")

        show = []
        for m in models:
            show.append((f"{m}", sub[sub.method == f"{m} (direct)"].mae_eval.iloc[0]))
        show.append(("mean of models", sub[sub.method == "mean of direct"].mae_eval.iloc[0]))
        show.append(("ensemble (per-dataset)", sub[sub.method.str.startswith("ENSEMBLE-per")].mae_eval.iloc[0]))
        show.append(("ensemble (joint)", sub[sub.method.str.startswith("ENSEMBLE-joint")].mae_eval.iloc[0]))

        labels = [s[0] for s in show]
        pct = [100.0 * (best - v) / best for _, v in show]
        colors = ["tab:green" if p > 0.05 else "tab:red" if p < -0.05 else "tab:gray" for p in pct]

        y = np.arange(len(labels))[::-1]
        ax.barh(y, pct, color=colors)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_yticks(y); ax.set_yticklabels(labels)
        for yi, p in zip(y, pct):
            ax.text(p, yi, f" {p:+.1f}% ", va="center",
                    ha="left" if p >= 0 else "right", fontsize=9)
        lo, hi = min(pct + [0]), max(pct + [0])
        pad = 0.25 * max(hi - lo, 1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_xlabel(f"% better than best single model ({best_name})")
        ax.set_title(nm)
    fig.suptitle("Did the ensemble beat the best single model?   green = better, red = worse")
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight")


def component_marginals(scores, models):
    comps = ["trend", "seasonal", "residual"]
    table = np.zeros((3, len(models)))
    for ci in range(3):
        for mi, m in enumerate(models):
            vals = [v for combo, v in scores.items() if combo[ci] == m]
            table[ci, mi] = np.mean(vals)
    return comps, table


def plot_component_table(joint, per_ds_sel, datasets, models, path):
    panels = [("joint (both datasets)", joint)] + [(d["name"], per_ds_sel[d["name"]]) for d in datasets]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.6), squeeze=False)
    axes = axes[0]
    for ax, (title, scores) in zip(axes, panels):
        comps, table = component_marginals(scores, models)
        norm = (table - table.min(axis=1, keepdims=True)) / (np.ptp(table, axis=1, keepdims=True) + 1e-12)
        ax.imshow(norm, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(models))); ax.set_xticklabels(models)
        ax.set_yticks(range(3)); ax.set_yticklabels(comps)
        for ci in range(3):
            best = np.argmin(table[ci])
            for mi in range(len(models)):
                txt = f"{table[ci, mi]:.3f}" + ("  \u2713" if mi == best else "")
                ax.text(mi, ci, txt, ha="center", va="center", fontsize=9,
                        fontweight="bold" if mi == best else "normal")
        ax.set_title(title)
    fig.suptitle("Which model is best at each component?   (avg score when that model handles it, lower = better, \u2713 = best)")
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()