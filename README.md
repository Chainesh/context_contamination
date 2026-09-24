# Controlled Context Contamination Analysis (CCCA)

An experimental framework for probing **how time-series foundation models use their input context**. Instead of only asking how accurate a model is, CCCA asks:

- How much of the context can be corrupted before the forecast breaks?
- Which parts of the context does the model actually rely on?
- Does the model behave like a linear operator (superposition, shift and scale equivariance)?
- If the context is decomposed into trend, seasonal, and residual, which component matters, and which model handles each component best?

Every experiment keeps the **forecast target fixed and clean** and modifies only the context, so any change in error is caused by the context manipulation alone.

---

## Contents

1. [Models](#models)
2. [Datasets](#datasets)
3. [Installation](#installation)
4. [Repository structure](#repository-structure)
5. [Experiments](#experiments)
   - [A. Context contamination](#a-context-contamination)
   - [B. Which context matters](#b-which-context-matters)
   - [C. Linearity and structure](#c-linearity-and-structure)
   - [D. Decomposition and ensembling](#d-decomposition-and-ensembling)
6. [Running everything](#running-everything)
7. [Preliminary observations](#preliminary-observations)
8. [Known caveats](#known-caveats)

---

## Models

| Name (`--models`) | Model | Backbone family | Input scale |
|---|---|---|---|
| `naive` | Repeat last value | Baseline | normalized |
| `naive_seasonal` | Repeat last H values | Baseline | normalized |
| `linear` | Ridge regression on the full context | Linear baseline | normalized |
| `ttm` | IBM Granite TTM r2 (512-96) | MLP-Mixer (TSMixer), RevIN | normalized |
| `timesfm` | Google TimesFM 1.0-200m | Patch decoder transformer | raw |
| `chronos` | Amazon Chronos-Bolt base | T5 encoder-decoder, value tokenization | raw |
| `tirex` | NX-AI TiRex | xLSTM (recurrent) | raw |
| `arima` | `pmdarima.auto_arima` | Statistical, refit per window | normalized (analytic test only) |

**Scale handling.** Data is z-normalized with training statistics. Models marked `raw` receive de-normalized input, and their output is re-normalized before scoring, so all MAE values are comparable across models.

All adapters live in `src/model_adapters.py` and share one interface.

```python
predictions = model.predict_batch(contexts, horizon)   # [B, L] -> [B, H]
```

---

## Datasets

| Dataset | Frequency | Target | Length | Seasonal period |
|---|---|---|---|---|
| ETTm1 | 15 min | `OT` (oil temperature) | ~69.7k | 96 (daily) |
| US Births | daily | `births` | 7,670 (1994 to 2014) | 7 (weekly) |

**Splits.** 70% train, 10% val, 20% test, in time order. Normalization uses train statistics only.

**Windowing.** Context length 512, horizon 96. Windows are cut from the test split with a configurable stride.

**Preparing US Births** (merges the fivethirtyeight CDC 1994 to 2003 and SSA 2000 to 2014 files)

```bash
python scripts/prepare_us_births.py
```

---

## Installation

Python 3.10 or newer, in a fresh virtual environment.

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt            # numpy<2, pandas, matplotlib, pyyaml, tqdm
pip install statsmodels pmdarima           # STL decomposition and auto-ARIMA
pip install torch
pip install "granite-tsfm @ git+https://github.com/ibm-granite/granite-tsfm.git"
pip install "transformers>=4.40,<4.47"     # TTM is not compatible with newer transformers
pip install timesfm                        # 1.x API (TimesFm, TimesFmHparams)
pip install chronos-forecasting
```

**TiRex** needs its own environment (it pins dependencies that conflict with the above) and an NVIDIA GPU with compute capability 8.0 or higher.

```bash
git clone https://github.com/NX-AI/tirex && cd tirex
conda env create --file requirements_py26.yaml
conda activate tirex && pip install -e .
```

Run TiRex experiments separately with `--models tirex` from that environment.

---

## Repository structure

```
ccca/
├── configs/
│   ├── default.yaml          # ETTm1
│   └── us_births.yaml        # US Births
├── data/raw/                 # ETTm1.csv, us_births.csv
├── src/
│   ├── data_loading.py       # load, split, train-only normalization
│   ├── windowing.py          # context / future windows
│   ├── contamination.py      # recency, span, distinguishability corruption
│   ├── model_adapters.py     # all models behind one interface
│   ├── evaluate.py           # contamination evaluation loop, tolerance metric
│   ├── metrics.py
│   ├── plotting.py
│   └── plotting_contamination.py
├── scripts/                  # one script per experiment (see below)
└── results/
    ├── tables/               # CSV outputs
    └── plots/                # PNG outputs (ensemble plots in plots/ensemble/)
```

Every script accepts `--config`, `--models`, `--device`, `--batch-size`, and a window limit (`--max-windows` or `--n-windows`).

---

## Experiments

### A. Context contamination

#### `run_pilot.py`

**Question.** How much corrupted context can a model tolerate before its error rises meaningfully?

**Method.** One contiguous block of the context is corrupted. It is controlled by three independent knobs.

| Knob | Range | Meaning |
|---|---|---|
| recency `r` | 0 to 1 | 0 means the block touches the forecast boundary, 1 means it touches the start of the context |
| span `s` | 0 to 1 | fraction of the context inside the block |
| distinguishability `d` | 0 to 1 | corruption strength inside the block (0 untouched, 1 maximal) |

Corruption mechanisms applied inside the block

```
additive_noise       x + d * std(block) * eps
signal_mixing        (1 - d) * x + d * donor
segment_replacement  a fraction d of block positions replaced by donor values
```

The donor is a different test window chosen with a fixed seed, and every model sees identical corruptions.

**Metric.** Context contamination tolerance, the largest `d` for which `MAE(d) <= 1.1 * MAE(0)`.

**Outputs.** `contamination_results.csv`, `context_contamination_tolerance.csv`, `contamination_coverage.png`, `contamination_example.png`, `degradation_*.png`, `recency_sensitivity_*.png`, `distinguishability_star_comparison.png`

```bash
python scripts/run_pilot.py --models naive ttm timesfm chronos --max-windows 200 --device cuda
```

---

### B. Which context matters

#### `run_recent_ablation.py`

**Question.** Does removing short segments from the recent context hurt the forecast?

**Method.** A contiguous segment of length k in {2, 4, 8, 16, 32} is removed from a region near the forecast boundary and filled by linear interpolation. The last `--protect-recent` points are never touched. The experiment is repeated over 5 seeds.

**Outputs.** `recent_removal_ablation.csv`, `recent_removal_ablation.png`, `removal_example_k*.png`

```bash
python scripts/run_recent_ablation.py --models ttm timesfm chronos --protect-recent 64 --device cuda
```

#### `run_positional_sensitivity.py`

**Question.** Where in the context does removal hurt most?

**Method.** A fixed-length removal window is slid across the entire context. The relative degradation `MAE / MAE_clean` is recorded at each position. The sweep is deterministic, so no seeds are needed.

**Outputs.** `positional_sensitivity.csv`, `positional_sensitivity.png`, `positional_sensitivity_with_context.png` (curve aligned under a real context window)

```bash
python scripts/run_positional_sensitivity.py --models ttm timesfm chronos --segment-lengths 16 32 --stride 32 --device cuda
```

#### `run_pip_prefix_flatten.py`

**Question.** How far back does real history need to extend?

**Method.** For each window, M Perceptually Important Points (PIP) are computed. Using each PIP point as a cutoff, everything before the cutoff is flattened to the cutoff value and everything after it is left intact. MAE is recorded per cutoff.

**Outputs.** `pip_prefix_flatten.csv`, `pip_prefix_flatten.png`, `pip_prefix_heatmap_{model}.png`

```bash
python scripts/run_pip_prefix_flatten.py --models ttm timesfm chronos --m 8 --device cuda
```

#### `run_pip_ablation.py`, `run_shape_region_ablation.py`

Removal of PIP-important versus PIP-unimportant versus random points, and shape-region ablations. Outputs `pip_side_by_side.png` and `shape_region_ablation.png`.

#### `run_worst_windows.py`

**Question.** Do different architectures fail on the same windows?

**Method.** Per-window MAE for each model. The top-N worst windows are compared across models using overlap and the Jaccard index. High overlap means the difficulty lives in the data, and low overlap means the failure is architecture specific.

**Outputs.** `per_window_mae.csv`, `worst_windows.png` (grid of context, truth, and forecasts), `worst_window_overlap.png`

```bash
python scripts/run_worst_windows.py --models ttm timesfm chronos --top-n 20 --device cuda
```

---

### C. Linearity and structure

A model `f` is linear if `f(a·x + b·y) = a·f(x) + b·f(y)`. Each test measures a **gap** between the two sides, averaged over the 96 forecast steps. A gap of zero means the property holds.

#### `run_linearity_test.py`

Four tests in one script.

| Test | Compares | Probes |
|---|---|---|
| Decomposition | `f(trend + residual)` vs `f(trend) + f(residual)` | superposition on a 2-part split |
| Seasonal (STL) | `f(T + S + R)` vs `f(T) + f(S) + f(R)` | superposition on an STL split |
| Offset | `f(x + c)` vs `f(x) + c` | shift equivariance (RevIN mean) |
| Scale | `f(a·x)` vs `a·f(x)` | scale equivariance (RevIN std) |

A model with reversible instance normalization (RevIN) drives both the offset and scale gaps to about zero.

**Outputs.** `linearity_test.csv`, `linearity_seasonal_per_window.csv`, `linearity_summary.png`, `linearity_decomposition_example.png`, `linearity_offset_example.png`, `linearity_seasonal_hist_raw.png`, `linearity_seasonal_hist_relative.png`, `linearity_seasonal_example.png`, `revin_check.png`

```bash
python scripts/run_linearity_test.py --models ttm timesfm chronos --period 96 --shift 1.0 --factor 2.0 --device cuda
```

#### `run_error_superposition.py`

**Additive, in error space.** For paired windows `x` and `y` with measured futures `Tx` and `Ty`, errors are taken against the linear-equivalent target `Tx + Ty`.

```
gap = | e(x+y) - (e(x) + e(y)) |
```

Algebraically the truth terms cancel, so this equals `|f(x+y) - f(x) - f(y)|`. The script reports a cancellation residual (expected about 1e-15) as a built-in correctness check.

**Multiplicative, in forecast space.** `|f(x·y) - f(x)·f(y)|`. This test is deliberately not framed in error space, because for multiplication the truth terms do not cancel, and the error framing would mix nonlinearity with plain forecast inaccuracy.

**Outputs.** `error_superposition.csv`, `error_superposition_hist.png`, `multiplicative_superposition_hist.png`

```bash
python scripts/run_error_superposition.py --models ttm timesfm chronos --device cuda
```

#### `run_linear_combination_sweep.py`

**Method.** A 9 x 9 grid of `(a, b)` values in [-1, 1].

```
gap(a, b) = | f(a·x + b·y) - (a·f(x) + b·f(y)) |
```

The point `(1, 1)` recovers the additive test. The lines `a = 0` and `b = 0` are trivially zero and act as a sanity check.

**Outputs.** `linear_combination_grid.csv`, `linear_combination_heatmap.png`

```bash
python scripts/run_linear_combination_sweep.py --models ttm timesfm chronos --grid-size 9 --max-windows 100 --device cuda
```

Cost is `grid_size^2 x windows` forecasts per model.

#### `run_analytic_superposition.py`

**Method.** Superposition gap on synthetic signals `t`, `t^2`, `sin(t)`, `cos(t)`, `cos(3t)` (each normalized), comparing auto-ARIMA against zero-shot foundation models. Reports per-function gaps and the 5-function average.

**Outputs.** `analytic_superposition.csv`, `analytic_superposition_hist.png`, `analytic_superposition_bars.png`

```bash
python scripts/run_analytic_superposition.py --models arima ttm timesfm chronos --n-windows 100 --device cuda
```

---

### D. Decomposition and ensembling

All scripts in this group use STL (`statsmodels.tsa.seasonal.STL`, robust) with the dataset's seasonal period. The trend, seasonal, and residual components sum exactly to the original context.

#### `run_stl_strategy.py`

**Question.** Is it more accurate to forecast the whole context, or to forecast each component and add the results?

```
MAE_direct     = | f(context)              - truth |
MAE_decomposed = | f(T) + f(S) + f(R)      - truth |
```

Reports mean and median MAE, the win rate (fraction of windows where decomposition wins), and the per-window difference.

**Outputs.** `stl_strategy_comparison.csv`, `stl_strategy_per_window.csv`, `stl_strategy_example.png` (with ground truth), `stl_strategy_mae.png`, `stl_strategy_diff_hist.png`

```bash
python scripts/run_stl_strategy.py --models ttm timesfm chronos --period 96 --device cuda
```

#### `run_component_contribution.py`

**Question.** How much does each component contribute to accuracy?

**Method.** Leave one component out and measure the increase in MAE. Two modes are reported.

| Mode | Removal |
|---|---|
| `sum` | drop the component's forecast from `f(T) + f(S) + f(R)` |
| `context` | drop the component from the context, then forecast directly |

```
contribution(C) = MAE(without C) - MAE(all)
```

**Outputs.** `component_contribution.csv`, `component_contribution.png`

```bash
python scripts/run_component_contribution.py --models ttm timesfm chronos --period 96 --device cuda
```

#### `run_component_ensemble.py`

**Question.** If each model is best at a different component, does combining them beat any single model?

**Method.**
1. Every model forecasts T, S, and R for every window.
2. All 27 model-to-component assignments are evaluated.
3. Test windows are split in time order. The first half is used to **select** an assignment, and the second half (held out) is used to **evaluate** it.
4. Two selection rules are compared.
   - **Per-dataset** picks the best assignment separately for each dataset.
   - **Joint** picks one assignment for all datasets together. Each dataset's MAE is divided by its best single-model MAE before averaging, so no dataset dominates because of its error scale.
5. Baselines are each model direct, each model decomposed, and the mean of the direct forecasts.

**Outputs** (plots in `results/plots/ensemble/`)

| File | Shows |
|---|---|
| `component_ensemble_multi.csv` | held-out MAE for every method and dataset |
| `component_ensemble_joint_combos.csv` | joint score for all 27 assignments |
| `ensemble_improvement.png` | % better or worse than the best single model |
| `ensemble_best_model_per_component.png` | best model for each component (3 x 3 table) |
| `ensemble_multi_dataset.png` | all methods ranked by held-out MAE |
| `ensemble_joint_combos.png` | heatmap of all 27 assignments |

```bash
python scripts/run_component_ensemble.py \
  --configs configs/default.yaml configs/us_births.yaml \
  --models ttm timesfm chronos --max-windows 200 --device cuda
```

The seasonal period is read from each config (`period: 96` for ETTm1, `period: 7` for US Births).

---

## Running everything

```bash
# save previous results first, since every script overwrites fixed filenames
cp -r results results_backup

M="ttm timesfm chronos"
python scripts/run_pilot.py                     --models $M --device cuda
python scripts/run_worst_windows.py             --models $M --device cuda
python scripts/run_positional_sensitivity.py    --models $M --device cuda
python scripts/run_pip_prefix_flatten.py        --models $M --device cuda
python scripts/run_recent_ablation.py           --models $M --device cuda
python scripts/run_linearity_test.py            --models $M --device cuda --period 96
python scripts/run_error_superposition.py       --models $M --device cuda
python scripts/run_linear_combination_sweep.py  --models $M --device cuda
python scripts/run_analytic_superposition.py    --models arima $M --device cuda
python scripts/run_stl_strategy.py              --models $M --device cuda --period 96
python scripts/run_component_contribution.py    --models $M --device cuda --period 96
python scripts/run_component_ensemble.py        --models $M --device cuda \
    --configs configs/default.yaml configs/us_births.yaml
```

For US Births, add `--config configs/us_births.yaml` and use `--period 7`.

---

## Preliminary observations

These are single-seed results on about 140 test windows per dataset and should be treated as preliminary.

- **Recency dominates.** All models are nearly insensitive to removal across most of the 512-step context, with sharp degradation only in the last 30 to 50 steps.
- **Accuracy and robustness trade off.** On ETTm1, Chronos has the lowest clean MAE but the largest degradation when recent context is removed, while TTM is least accurate among the foundation models but most robust.
- **Shared failure modes.** 12 of the 20 worst windows are shared by TTM and TimesFM, mostly abrupt regime changes at the forecast boundary.
- **Linearity.** TTM consistently shows the smallest superposition gaps. TTM and TimesFM are near-perfectly shift and scale equivariant, while Chronos shows a measurable spread.
- **Component weakness.** On US Births, TTM fails specifically on the seasonal component, which explains its large error on that dataset.
- **Ensembling.** Ensembles help on ETTm1 (simple averaging is best, about 4% better than Chronos) and do not beat Chronos on US Births, where one model dominates.

---

## Known caveats

- **Overwriting.** Output filenames are fixed. Copy `results/` between runs on different datasets or environments.
- **Out-of-distribution inputs.** Combined signals (`x + y`, `a·x + b·y`, isolated STL components) are inputs no model saw in training, so part of any measured gap reflects distribution shift rather than pure nonlinearity.
- **ARIMA is not a fixed linear operator.** Auto-ARIMA refits and reselects its order for every input, so its superposition gap reflects refitting instability. It is also run with `seasonal=False`, which handicaps it on sinusoidal signals.
- **Normalization inside models.** Large decomposition gaps partly come from each component being normalized differently inside the model, not only from nonlinear layers.
- **Heatmap colors.** In `ensemble_best_model_per_component.png`, colors are scaled per row, so small differences can look large. Read the numbers.
- **Held-out split.** The ensemble selection and evaluation halves are split in time order, so held-out results also include any drift within the test period.
- **Checkpoint constraints.** The TTM checkpoint is tied to context 512 and horizon 96. TimesFM 1.0 is used rather than 2.5 because of API compatibility.
