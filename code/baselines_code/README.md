# Adaptive Hybrid Kernel SVMs — Baselines & Experimental Harness

Reference implementation of the **baseline models** and the **experimental
harness** for the Master's thesis:

> *"Adaptive Hybrid Kernel SVMs: Dynamic Feature Weighting for Noisy,
> High-Dimensional IoT Healthcare Data."*

The proposed model (Adaptive Hybrid Kernel SVM with dynamic per-feature
weighting) is included as a **stub** that plugs into the same harness; only its
weight-learning `fit()` body is left as a TODO (the core thesis contribution).

## What is implemented

**Baselines** (isolate the contribution of the hybrid kernel + feature weights):
1. **SVM-RBF** — standard RBF SVM; `(C, gamma)` tuned by CV; no feature weights.
2. **SVM-Linear** — standard linear SVM; `C` tuned by CV; no feature weights.
3. **Static MKL** — convex combination `K = mu*K_linear + (1-mu)*K_rbf`,
   `mu` tuned by CV, via `SVC(kernel='precomputed')`; **no feature weights**.
   This is the key ablation (hybrid kernel *without* weighting).

**Proposed (stub):** `AdaptiveHybridKernelSVM` — same interface; reuses
`combined_kernel(..., feature_weights=w)`. Its `fit()` raises
`NotImplementedError` until you implement the weighting logic.

**Harness** (`evaluation.py`):
- 5-fold **stratified** CV for hyperparameter tuning on the train split.
- Whole experiment repeated 5× with different seeds → **mean ± std**.
- Metrics: **Accuracy** and **Macro-F1**.
- **Noise robustness:** Gaussian noise with `sigma ∈ {0.1,0.3,0.5,0.7,1.0}`
  injected into a random **30%** of feature dimensions **at test time only** →
  degradation curves (CSV + PNG).
- **Ablation:** any registered set of models can be compared (full vs Static MKL).
- **Significance:** paired **Wilcoxon signed-rank** test across the
  5×5 = 25 fold-level results, proposed vs each baseline (p < 0.05 threshold).

## Project layout

```
baselines_code/
├── data.py             # Dataset container, synthetic generator, real loaders (TODO)
├── preprocessing.py    # imputation + standardization + noise/dropout utilities
├── kernels.py          # linear / RBF / static-MKL combined Gram matrices
├── models.py           # 3 baselines + proposed-model stub (common interface)
├── evaluation.py       # CV tuning, repeated runs, metrics, noise curves, Wilcoxon
├── run_experiments.py  # main entry point
├── requirements.txt
├── README.md
├── data/               # drop real datasets here (opportunity/, casas/)
└── results/            # CSVs + PNG written here on run
```

## How to run (out of the box, synthetic data)

```bash
pip install -r requirements.txt
python run_experiments.py
```

Useful flags:

```bash
python run_experiments.py --n-samples 600 --n-features 80 --n-repeats 5
python run_experiments.py --include-proposed   # tries the stub (gets skipped)
```

Results are written to `results/`:
`summary_results.csv`, `noise_curves.csv`, `wilcoxon_results.csv`,
`noise_curves.png`.

## Dropping in the real datasets

Place raw files under `data/` and implement the two loaders in `data.py`.

### OPPORTUNITY (Chavarriaga et al., 2013)
- Path: `data/opportunity/` (raw `S*-ADL*.dat`, space-separated).
- ~113 sensor columns (body-worn IMUs + ambient) + label columns.
- TODO in `load_opportunity()`: read `.dat`, select the 113 feature columns,
  integer-encode the chosen label (e.g. Locomotion), optionally drop NULL class.

### CASAS smart-home (Cook et al., 2013)
- Path: `data/casas/` (raw event logs: `DATE TIME SENSOR MESSAGE [LABEL]`).
- Ambient motion/door/temperature sensors; asynchronous events.
- TODO in `load_casas()`: window the events, build per-window feature vectors,
  assign dominant activity label, integer-encode.

Both return a `Dataset(X, y, name=...)`; everything downstream is unchanged.
Then run, e.g.: `python run_experiments.py --dataset opportunity`.

## Implementing the proposed model

In `models.py::AdaptiveHybridKernelSVM.fit`, learn per-feature weights
`self.feature_weights_` (e.g. via kernel-target alignment or alternating
optimization), fit `SVC(kernel='precomputed')` on
`combined_kernel(X, mu=, gamma=, feature_weights=self.feature_weights_)`, and
store `self._X_train`. The `predict()` path and the harness already support it,
including the Wilcoxon comparison against the baselines.

## Reproducibility
All randomness is seeded (`--seed`, default 42); each repeat uses `seed + r`.
