# Experimental Codebase — Overview

This folder contains the experimental stand for the thesis **"Adaptive Hybrid
Kernel SVMs: Dynamic Feature Weighting for Noisy, High-Dimensional IoT
Healthcare Data."** It takes data, cleans it, trains the SVM models, measures
their accuracy and robustness to noise, and compares them statistically.

The pipeline runs end-to-end **right now** on synthetic data. The two real
benchmarks (OPPORTUNITY and CASAS) have documented loader stubs that still need
to be implemented (this is the remaining part of Phase 1).

Code lives in `baselines_code/`: `data.py`, `preprocessing.py`, `kernels.py`,
`models.py`, `evaluation.py`, `run_experiments.py`.

---

## Preprocessing (`preprocessing.py`)

Raw sensor data cannot be fed to an SVM directly. The `Preprocessor` class does
two things:

1. **Missing-value imputation.** IoT sensors regularly go silent, leaving gaps
   (NaN). The imputer fills these, e.g. with the feature mean, so the model does
   not break on empty cells.
2. **Standardization.** Different sensors live on different scales (heart rate
   ~60–100, accelerometer ~0–1), which confuses an SVM. Standardization rescales
   each feature to mean 0 and unit variance so all features are comparable.

A key correctness detail: preprocessing is **fit on the training fold only** and
then applied to the test fold. Computing statistics over all data (including the
test set) would be *data leakage* and would inflate results. We avoid this.

## Noise injection (`preprocessing.py` → `inject_gaussian_noise`)

This is the core of hypothesis **H2 (noise resilience)**. To show the model is
more robust to sensor degradation, we have to simulate that degradation. The
function takes the test data and **deliberately corrupts it**: it adds Gaussian
noise to a random 30% of features at increasing strengths (sigma = 0.1, 0.3,
0.5, 0.7, 1.0). This mimics sensors starting to drift, lie, or become noisy.
`simulate_dropout` additionally models complete sensor failure (sets values to
NaN).

We then run each model on progressively noisier data and plot **degradation
curves** (accuracy falls as noise rises). If the adaptive model's curve falls
more slowly than a standard SVM's, H2 is supported.

## Harness (`evaluation.py` + `run_experiments.py`)

A "harness" is the wiring that **runs all the experiments automatically**. It is
not a model; it is the conveyor that:

- runs 5-fold **stratified** cross-validation to tune hyperparameters;
- repeats the whole experiment 5 times with different seeds (so the result is
  stable, not luck) and reports **mean ± std**;
- computes the metrics (accuracy, macro-F1);
- builds the noise-robustness curves;
- runs the **ablation** (full model vs. the weight-free variant);
- tests statistical significance with a paired **Wilcoxon** test (p < 0.05).

You configure it once, then ask it to "run all models on this dataset" and get
ready-made tables and figures for the Results chapter.

## How the datasets are used (`data.py`)

`load_dataset("opportunity")` returns data in a single format (a `Dataset`
object with feature matrix `X` and labels `y`), so the harness runs every model
on it the same way. Currently only `"synthetic"` is available
(`make_synthetic_iot_dataset` — generates fake high-dimensional, imbalanced data
with noise and dropout so the pipeline can be tested today). The real
`"opportunity"` and `"casas"` loaders are stubs that still need implementing.

### OPPORTUNITY (Chavarriaga et al., 2013, *Pattern Recognition Letters*, Elsevier)

A public activity-recognition benchmark. People in a room wore many body sensors
(accelerometers, gyroscopes) plus object sensors and performed daily actions
(cooking, opening doors, etc.). About **113 features**, with natural sensor
dropout. For this thesis it is a stress test of high dimensionality and
realistic noise.

### CASAS (Cook et al., 2013, *IEEE Computer*)

A "smart home in a box": motion, door, and temperature sensors in real homes,
monitoring the activity of older adults. The data is a different type — not a
ready table but a **stream of asynchronous events** ("at 10:05 motion sensor M12
fired"). This matches the ambient elderly-monitoring motivation of the thesis.

Two deliberately different datasets are used to show the method works across
different kinds of IoT-healthcare data, not just one narrow case.

## Feature extraction pipelines

An SVM expects a **table**: one row per sample, columns = features. OPPORTUNITY
is already close to this — you just select the ~113 sensor columns and encode the
activity label. The simple case.

CASAS is a **raw event log**, not a table. You cannot feed "sensor M12 fired"
into an SVM. A **feature extraction pipeline** turns the event stream into a
table: cut the stream into **windows** (e.g., every N events or T seconds) and,
for each window, compute features — how many times each sensor fired, which fired
last, durations, etc. That gives one feature vector per window, plus the dominant
activity label. This "windowing + aggregation into a feature vector" is feature
extraction.

The two loaders (`load_opportunity`, `load_casas`) currently hold these steps as
documented TODO stubs. Implementing them against the real files completes
Phase 1.

---

## Phase 1 status (Data Preparation & Literature Finalization)

| Deliverable | Status |
|---|---|
| Finalize literature review chapter | Done (draft) |
| Preprocessing code | Done (`preprocessing.py`) |
| Noise injection module | Done (`inject_gaussian_noise`) |
| OPPORTUNITY: data + loader + pipeline run | Done — `load_opportunity` implemented; runs on real data |
| CASAS: data + loader + pipeline run | Done — `load_casas` implemented; runs on real data |

Both datasets are fully wired and the harness runs end-to-end on each:

    python run_experiments.py --dataset opportunity --max-samples 1500
    python run_experiments.py --dataset casas --max-samples 1500

OPPORTUNITY: the UCI archive is unpacked under `baselines_code/data/opportunity/`;
`load_opportunity` parses the `.dat` files (242 sensor channels, Locomotion
labels, real sensor NaNs, stratified subsampling for tractability).

CASAS: the Zenodo record is unpacked under `baselines_code/data/casas/labeled/`;
`load_casas` reads one home's event log (default `hh101.csv`), forward-fills the
activity through begin/end markers, and windows the event stream into per-sensor
activation-count features plus a cyclic hour-of-day encoding. The chosen record
is room-level (6 zones -> 8 features); a raw-sensor CASAS home can be swapped in
for a higher-dimensional variant. `top_k_activities` keeps the most frequent
classes (default 12) for a clean multi-class problem.

**Phase 1 is complete.** Phase 2 is the core model: implement
`AdaptiveHybridKernelSVM.fit` in `models.py` (per-feature data-quality weights
inside the hybrid kernel); the harness, ablation, and Wilcoxon test are already
wired for it.
