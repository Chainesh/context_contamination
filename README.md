# Controlled Context Contamination Analysis (CCCA)

Measures how much of a forecasting model's input context can be corrupted
before its predictions degrade. The context is corrupted, the forecast
target is never touched, and the drop in accuracy is measured as a
function of three independent knobs:

- **recency** — how close the corrupted block sits to the forecast boundary
  (0 = touches the last observed timestep, 1 = touches the start of context)
- **span** — what fraction of the context the block covers
- **distinguishability** — how far the block's values are pushed from the
  true signal (0 = untouched, 1 = fully replaced or max noise)

`distinguishability*` (the largest tolerated distinguishability before
error exceeds a chosen tolerance over the clean baseline) is the main
robustness number this framework produces, per model and per
(mechanism, recency, span) setting.

## Install

Base pipeline (naive, seasonal, linear baselines), no GPU or heavy deps:

```bash
pip install -r requirements.txt
```

Foundation models (TTM, TimesFM). Everything runs on CPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-models.txt
```

TTM lives in the `tsfm_public` namespace from the pinned `granite-tsfm`
tag, not in mainline transformers. Both TTM and TimesFM auto-detect
device and fall back to CPU. TTM is ~1M params and forecasts in well
under a second per window on CPU. TimesFM is 200M params, so expect a
minute or two of model load plus slower per-batch inference, still fine
on CPU for a pilot.

## Data

`data/raw/ETTm1.csv` is included. Point `configs/default.yaml`'s
`data.path` / `data.target_col` elsewhere to use a different series.

## Run

Baselines only:

```bash
python scripts/run_pilot.py --config configs/default.yaml \
    --models naive naive_seasonal linear --max-windows 200
```

Add the foundation models (CPU is the default device now):

```bash
python scripts/run_pilot.py --config configs/default.yaml \
    --models naive linear ttm timesfm --max-windows 200 --device cpu
```

## Outputs

- `results/tables/contamination_results.csv` — one row per (model,
  mechanism, recency, span, distinguishability) with mae/mse/rmse/smape,
  clean_metric, relative_degradation, percent_increase
- `results/tables/context_contamination_tolerance.csv` — one row per
  (model, mechanism, recency, span) with distinguishability_star
- `results/plots/contamination_coverage.png` — which context positions
  get touched across the recency sweep, one panel per span
- `results/plots/contamination_example.png` — clean vs. corrupted values
  for one real window, sanity check for the contamination math
- `results/plots/degradation_{mechanism}_span{s}.png` — error vs.
  distinguishability, one curve per (model, recency)
- `results/plots/recency_sensitivity_{mechanism}_span{s}.png` — whether
  a model depends more on recent or distant context
- `results/plots/distinguishability_star_comparison.png` — tolerance bar
  chart across models and settings

## Reading distinguishability*

Higher is more robust. If TimesFM has lower clean MAE but a lower
`distinguishability*` than TTM, TimesFM is more accurate on clean
context but breaks sooner as that context gets corrupted. If a model's
tolerance drops sharply as recency approaches 0, its forecasts lean
heavily on the most recent observations rather than the full window.
