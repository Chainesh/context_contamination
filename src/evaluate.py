import sys
import numpy as np
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from contamination import ContaminationParams, apply_contamination_batch, choose_donor_indices
from metrics import compute_metrics, relative_degradation, percent_increase


def evaluate_model_under_contamination(model, contexts, futures, mean, std, donor_indices,
                                        mechanisms, recency_values, span_values,
                                        distinguishability_values, batch_size, seed):
    rng = np.random.default_rng(seed)
    N, L = contexts.shape
    H = futures.shape[1]
    donor_contexts_all = contexts[donor_indices]
    rows = []

    for mechanism in mechanisms:
        for recency in recency_values:
            for span in span_values:
                for d in distinguishability_values:
                    params = ContaminationParams(recency=recency, span=span, distinguishability=d, mechanism=mechanism)
                    preds = []
                    for start in range(0, N, batch_size):
                        end = start + batch_size
                        batch_contexts = contexts[start:end]
                        batch_donors = donor_contexts_all[start:end]
                        contaminated = apply_contamination_batch(batch_contexts, batch_donors, params, rng)

                        model_input = contaminated * std + mean if model.scale == "raw" else contaminated
                        pred = model.predict_batch(model_input, H)
                        if model.scale == "raw":
                            pred = (pred - mean) / std
                        preds.append(pred)

                    preds = np.concatenate(preds, axis=0)
                    row = {
                        "model": model.name,
                        "mechanism": mechanism,
                        "recency": recency,
                        "span": span,
                        "distinguishability": d,
                        **compute_metrics(futures, preds),
                        "num_windows": N,
                        "context_length": L,
                        "horizon": H,
                    }
                    rows.append(row)

    return pd.DataFrame(rows)


GROUP_KEYS = ["model", "mechanism", "recency", "span"]


def add_relative_degradation(df: pd.DataFrame, primary_metric: str = "mae") -> pd.DataFrame:
    clean = df[df["distinguishability"] == 0.0][GROUP_KEYS + [primary_metric]].rename(
        columns={primary_metric: "clean_metric"}
    )
    out = df.merge(clean, on=GROUP_KEYS, how="left")
    out["relative_degradation"] = out[primary_metric] / out["clean_metric"]
    out["percent_increase"] = 100.0 * (out[primary_metric] - out["clean_metric"]) / out["clean_metric"]
    return out


def compute_context_contamination_tolerance(df: pd.DataFrame, primary_metric: str = "mae",
                                             tolerance_percent: float = 10.0) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(GROUP_KEYS):
        clean_row = group[group["distinguishability"] == 0.0]
        if clean_row.empty:
            continue
        clean_metric = clean_row[primary_metric].iloc[0]
        threshold = clean_metric * (1 + tolerance_percent / 100.0)
        within = group[group[primary_metric] <= threshold]
        d_star = within["distinguishability"].max() if not within.empty else 0.0
        row = dict(zip(GROUP_KEYS, keys))
        row.update({
            "primary_metric": primary_metric,
            "clean_metric": clean_metric,
            "tolerance_percent": tolerance_percent,
            "distinguishability_star": d_star,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def run_sanity_checks(df: pd.DataFrame, primary_metric: str = "mae") -> bool:
    passed = True

    for keys, group in df.groupby(GROUP_KEYS):
        if not (group["distinguishability"] == 0.0).any():
            print(f"[sanity] missing distinguishability=0 row for {dict(zip(GROUP_KEYS, keys))}")
            passed = False

    naive = df[df["model"] == "naive"]
    if not naive.empty:
        for mechanism in naive["mechanism"].unique():
            for span in naive["span"].unique():
                sub = naive[(naive["mechanism"] == mechanism) & (naive["span"] == span)]
                d_max = sub["distinguishability"].max()
                near = sub[(sub["recency"] == sub["recency"].min()) & (sub["distinguishability"] == d_max)]
                far = sub[(sub["recency"] == sub["recency"].max()) & (sub["distinguishability"] == d_max)]
                if not near.empty and not far.empty:
                    if not (near[primary_metric].iloc[0] > far[primary_metric].iloc[0]):
                        print(f"[sanity] expected recency=0 to hurt naive more than recency=1 "
                              f"for mechanism={mechanism} span={span}, got "
                              f"{near[primary_metric].iloc[0]:.4f} <= {far[primary_metric].iloc[0]:.4f}")
                        passed = False

    for model in df["model"].unique():
        clean = df[(df["model"] == model) & (df["distinguishability"] == 0.0)][primary_metric]
        if clean.std() / (clean.mean() + 1e-8) > 0.05:
            print(f"[sanity] clean {primary_metric} not consistent across settings for model={model}: "
                  f"mean={clean.mean():.4f} std={clean.std():.4f}")
            passed = False

    return passed
