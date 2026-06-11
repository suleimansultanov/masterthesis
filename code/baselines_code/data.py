"""
data.py
=======
Data loading utilities for the thesis:
"Adaptive Hybrid Kernel SVMs: Dynamic Feature Weighting for Noisy,
 High-Dimensional IoT Healthcare Data".

Provides:
  - A documented loader interface for the two public IoT-healthcare benchmarks
    (OPPORTUNITY and CASAS). These expect the real dataset files to be placed
    in a `data/` directory. The parsing bodies are intentionally left as
    clearly-marked TODOs because the raw files are not redistributed here.
  - A synthetic high-dimensional, multi-class, imbalanced dataset generator
    (with injectable feature noise and missing-value / sensor-dropout
    simulation) so the whole pipeline runs end-to-end out of the box.

All loaders return a `Dataset` object with `X` (float ndarray, shape
[n_samples, n_features]) and `y` (int ndarray, shape [n_samples]).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.datasets import make_classification


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    """Simple container for a loaded dataset."""

    X: np.ndarray
    y: np.ndarray
    name: str = "unnamed"
    feature_names: Optional[list] = field(default=None)

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y).astype(int)
        assert self.X.ndim == 2, "X must be 2D [n_samples, n_features]"
        assert self.X.shape[0] == self.y.shape[0], "X / y length mismatch"

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_classes(self) -> int:
        return int(len(np.unique(self.y)))

    def summary(self) -> str:
        classes, counts = np.unique(self.y, return_counts=True)
        dist = ", ".join(f"{c}:{n}" for c, n in zip(classes, counts))
        return (
            f"Dataset '{self.name}': {self.n_samples} samples, "
            f"{self.n_features} features, {self.n_classes} classes "
            f"(class distribution -> {dist})"
        )


# ---------------------------------------------------------------------------
# Synthetic generator (runs out of the box)
# ---------------------------------------------------------------------------
def make_synthetic_iot_dataset(
    n_samples: int = 400,
    n_features: int = 60,
    n_informative: int = 20,
    n_redundant: int = 10,
    n_classes: int = 4,
    class_weights: Optional[list] = None,
    dropout_rate: float = 0.05,
    base_noise: float = 0.0,
    random_state: int = 0,
) -> Dataset:
    """Generate a synthetic high-dimensional, multi-class, imbalanced dataset
    that mimics noisy IoT-healthcare sensor data.

    Parameters
    ----------
    n_samples : int
        Number of samples (kept small by default so the pipeline runs fast).
    n_features : int
        Total feature dimensionality (high-dimensional sensor space).
    n_informative, n_redundant : int
        Passed through to ``make_classification``. Remaining features are noise.
    n_classes : int
        Number of activity / state classes.
    class_weights : list or None
        Per-class proportions to create imbalance. If None, an imbalanced
        default is generated automatically.
    dropout_rate : float
        Fraction of entries set to NaN to simulate sensor dropout / missing
        values (handled later in preprocessing).
    base_noise : float
        Std of additional Gaussian noise added to every feature at generation
        time (separate from the test-time robustness noise).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    Dataset
    """
    rng = np.random.RandomState(random_state)

    if class_weights is None:
        # Build an imbalanced but non-degenerate distribution.
        raw = np.linspace(1.0, 0.3, n_classes)
        class_weights = list(raw / raw.sum())

    # make_classification requires informative+redundant+repeated <= n_features.
    n_repeated = 0
    assert n_informative + n_redundant + n_repeated <= n_features, (
        "n_informative + n_redundant must not exceed n_features"
    )

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=n_repeated,
        n_classes=n_classes,
        n_clusters_per_class=1,
        weights=class_weights,
        flip_y=0.01,            # small label noise
        class_sep=1.0,
        scale=rng.uniform(0.5, 5.0, size=n_features),  # heterogeneous sensor scales
        random_state=random_state,
    )

    if base_noise > 0:
        X = X + rng.normal(0.0, base_noise, size=X.shape)

    # Simulate sensor dropout / missing values as NaNs.
    if dropout_rate > 0:
        mask = rng.rand(*X.shape) < dropout_rate
        X[mask] = np.nan

    return Dataset(X=X, y=y, name="synthetic_iot")


# ---------------------------------------------------------------------------
# Real-dataset loaders (interfaces + TODO stubs)
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_opportunity(data_dir: str = DEFAULT_DATA_DIR) -> Dataset:
    """Loader for the OPPORTUNITY activity-recognition dataset.

    Reference
    ---------
    Chavarriaga et al. (2013), "The Opportunity challenge: A benchmark database
    for on-body sensor-based activity recognition", Pattern Recognition Letters.

    Expected files (NOT redistributed here) under ``data_dir/opportunity/``:
        S1-ADL1.dat, S1-ADL2.dat, ... (space-separated columns)
    Format notes:
        * ~113 sensor columns (body-worn IMUs + ambient sensors) + label cols.
        * Missing values are encoded; sensor dropout is natural in this data.
        * Use the "Locomotion" or "Mid-level gesture" column as the label,
          depending on the experiment.

    TODO (user must fill in):
        1. Read the .dat files (numpy.loadtxt / pandas.read_csv, sep=' ').
        2. Select the 113 sensor feature columns.
        3. Choose and integer-encode the label column.
        4. Drop / re-map the NULL (class 0) activity if desired.
        5. Return Dataset(X=..., y=..., name="opportunity").
    """
    folder = os.path.join(data_dir, "opportunity")
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"OPPORTUNITY data not found at '{folder}'. Place the raw .dat files "
            f"there and implement the parsing in load_opportunity(). "
            f"For a runnable demo use make_synthetic_iot_dataset() instead."
        )
    raise NotImplementedError(
        "OPPORTUNITY parsing is a TODO. See the docstring for the expected "
        "file format and the steps to implement."
    )


def load_casas(data_dir: str = DEFAULT_DATA_DIR) -> Dataset:
    """Loader for the CASAS smart-home dataset.

    Reference
    ---------
    Cook et al. (2013), "CASAS: A Smart Home in a Box", IEEE Computer.

    Expected files (NOT redistributed here) under ``data_dir/casas/``:
        raw sensor event logs, typically lines of:
            DATE TIME SENSOR_ID MESSAGE [ACTIVITY_LABEL]
    Format notes:
        * Ambient motion (M), door (D), and temperature (T) sensors.
        * Events are asynchronous; you must window/aggregate them into
          fixed-length feature vectors (e.g., sensor activation counts per
          window) before classification.

    TODO (user must fill in):
        1. Parse the event log lines.
        2. Segment events into windows (sliding window of N events or T sec).
        3. Build per-window feature vectors (counts / last-fired / durations).
        4. Assign each window the dominant activity label and integer-encode.
        5. Return Dataset(X=..., y=..., name="casas").
    """
    folder = os.path.join(data_dir, "casas")
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"CASAS data not found at '{folder}'. Place the raw event logs "
            f"there and implement the parsing in load_casas(). "
            f"For a runnable demo use make_synthetic_iot_dataset() instead."
        )
    raise NotImplementedError(
        "CASAS parsing is a TODO. See the docstring for the expected "
        "file format and the steps to implement."
    )


# Registry so the harness can request a dataset by name.
DATASET_LOADERS = {
    "synthetic": lambda **kw: make_synthetic_iot_dataset(**kw),
    "opportunity": lambda **kw: load_opportunity(**kw),
    "casas": lambda **kw: load_casas(**kw),
}


def load_dataset(name: str, **kwargs) -> Dataset:
    """Convenience dispatcher used by run_experiments.py."""
    if name not in DATASET_LOADERS:
        raise KeyError(f"Unknown dataset '{name}'. Options: {list(DATASET_LOADERS)}")
    return DATASET_LOADERS[name](**kwargs)
