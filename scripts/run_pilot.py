import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from utils import set_seed, load_config
from data_loading import load_univariate_csv, train_val_test_split, normalize_with_train_stats
from windowing import make_windows
from contamination import choose_donor_indices
from model_adapters import get_model
from evaluate import evaluate_model_under_contamination, add_relative_degradation, \
    compute_context_contamination_tolerance, run_sanity_checks
from plotting import plot_degradation_vs_distinguishability, plot_recency_sensitivity, \
    plot_distinguishability_star_comparison
from plotting_contamination import plot_contamination_coverage, plot_contamination_example


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    root = os.path.join(os.path.dirname(__file__), "..")
    cfg = load_config(os.path.join(root, args.config))

    if args.max_windows is not None:
        cfg["data"]["max_windows"] = args.max_windows
    if args.models is not None:
        cfg["models"]["names"] = args.models
    if args.device is not None:
        cfg["experiment"]["device"] = args.device
    if args.batch_size is not None:
        cfg["experiment"]["batch_size"] = args.batch_size

    set_seed(cfg["experiment"]["seed"])

    data_path = os.path.join(root, cfg["data"]["path"])
    series = load_univariate_csv(data_path, cfg["data"]["target_col"], cfg["data"]["time_col"])
    train, val, test = train_val_test_split(
        series, cfg["data"]["train_fraction"], cfg["data"]["val_fraction"], cfg["data"]["test_fraction"]
    )
    train_n, val_n, test_n, mean, std = normalize_with_train_stats(train, val, test)

    contexts, futures = make_windows(
        test_n, cfg["data"]["context_length"], cfg["data"]["horizon"],
        stride=cfg["data"]["stride"], max_windows=cfg["data"]["max_windows"],
    )
    train_contexts, train_futures = make_windows(
        train_n, cfg["data"]["context_length"], cfg["data"]["horizon"],
        stride=cfg["data"].get("train_stride", 48), max_windows=None,
    )

    rng = np.random.default_rng(cfg["experiment"]["seed"])
    donor_indices = choose_donor_indices(len(contexts), rng)

    cont_cfg = cfg["contamination"]
    plots_dir = os.path.join(root, "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_contamination_coverage(
        context_length=cfg["data"]["context_length"],
        span_values=cont_cfg["span_values"],
        save_path=os.path.join(plots_dir, "contamination_coverage.png"),
    )
    plot_contamination_example(
        contexts[0], contexts[donor_indices[0]],
        recency=0.0, span=cont_cfg["span_values"][0], distinguishability=0.8,
        mechanism="signal_mixing", rng=np.random.default_rng(0),
        save_path=os.path.join(plots_dir, "contamination_example.png"),
    )

    all_results = []
    for model_name in cfg["models"]["names"]:
        print(f"evaluating model: {model_name}")
        model = get_model(model_name, cfg, cfg["experiment"]["device"], cfg["experiment"]["batch_size"],
                          train_contexts=train_contexts, train_futures=train_futures)
        df = evaluate_model_under_contamination(
            model, contexts, futures, mean, std, donor_indices,
            cont_cfg["mechanisms"], cont_cfg["recency_values"], cont_cfg["span_values"],
            cont_cfg["distinguishability_values"], cfg["experiment"]["batch_size"], cfg["experiment"]["seed"],
        )
        all_results.append(df)

    results = pd.concat(all_results, ignore_index=True)
    results = add_relative_degradation(results, cfg["metrics"]["primary_metric"])
    cct = compute_context_contamination_tolerance(
        results, cfg["metrics"]["primary_metric"], cfg["metrics"]["tolerance_percent"]
    )

    tables_dir = os.path.join(root, "results", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    results.to_csv(os.path.join(tables_dir, "contamination_results.csv"), index=False)
    cct.to_csv(os.path.join(tables_dir, "context_contamination_tolerance.csv"), index=False)

    for mechanism in cont_cfg["mechanisms"]:
        for span in cont_cfg["span_values"]:
            plot_degradation_vs_distinguishability(
                results, mechanism, span, cfg["metrics"]["primary_metric"],
                save_path=os.path.join(plots_dir, f"degradation_{mechanism}_span{span}.png"),
            )
            plot_recency_sensitivity(
                results, mechanism, span, cfg["metrics"]["primary_metric"],
                save_path=os.path.join(plots_dir, f"recency_sensitivity_{mechanism}_span{span}.png"),
            )
    plot_distinguishability_star_comparison(cct, save_path=os.path.join(plots_dir, "distinguishability_star_comparison.png"))

    sanity_passed = run_sanity_checks(results, cfg["metrics"]["primary_metric"])

    print()
    print("=" * 60)
    print(f"windows: {len(contexts)}   context_length: {cfg['data']['context_length']}   horizon: {cfg['data']['horizon']}")
    print(f"models: {cfg['models']['names']}")
    print(f"mechanisms: {cont_cfg['mechanisms']}")
    print(f"recency_values: {cont_cfg['recency_values']}   span_values: {cont_cfg['span_values']}")
    print(f"best distinguishability*: {cct['distinguishability_star'].max():.3f}")
    print(f"worst distinguishability*: {cct['distinguishability_star'].min():.3f}")
    print(f"sanity checks passed: {sanity_passed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
