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


# OPPORTUNITY layout: 250 space-separated columns per row.
#   col 1 (index 0)        : MILLISEC timestamp
#   cols 2-243 (idx 1-242) : 242 sensor channels (body-worn / object / ambient)
#   col 244 (index 243)    : Locomotion label   {0=Null,1=Stand,2=Walk,4=Sit,5=Lie}
#   col 245 (index 244)    : HL_Activity label  {0,101..105}
#   col 250 (index 249)    : ML_Both_Arms (mid-level gesture) label
_OPP_SENSOR_SLICE = slice(1, 243)         # 242 sensor feature columns
_OPP_LABEL_COL = {                        # 0-based index of the chosen label
    "locomotion": 243,
    "hl_activity": 244,
    "ml_gesture": 249,
}


def load_opportunity(
    data_dir: str = DEFAULT_DATA_DIR,
    label: str = "locomotion",
    drop_null: bool = True,
    row_stride: int = 3,
    max_samples: Optional[int] = 5000,
    random_state: int = 0,
) -> Dataset:
    """Loader for the OPPORTUNITY activity-recognition dataset.

    Reference
    ---------
    Chavarriaga et al. (2013), "The Opportunity challenge: A benchmark database
    for on-body sensor-based activity recognition", Pattern Recognition Letters.

    Expects the unpacked archive under ``data_dir/opportunity/`` (the loader
    searches recursively for ``S*.dat`` files, so the nested
    ``OpportunityUCIDataset/dataset/`` folder is fine).

    Parameters
    ----------
    label : {"locomotion", "hl_activity", "ml_gesture"}
        Which annotation track to predict. "locomotion" (Stand/Walk/Sit/Lie)
        is the cleanest default; "ml_gesture" is the harder challenge task.
    drop_null : bool
        Drop rows whose label is 0 (the unlabelled / Null class).
    row_stride : int
        Keep every ``row_stride``-th frame to downsample the ~30 Hz stream
        (raw data is ~1.2M rows, far too many for a kernel SVM).
    max_samples : int or None
        Cap the final number of samples via a stratified random subsample, so
        the precomputed-kernel SVM stays tractable. None keeps everything.
    random_state : int
        Seed for the subsample.

    Returns
    -------
    Dataset
        X = sensor features (NaNs preserved for the imputer to handle),
        y = integer-encoded labels.
    """
    import glob
    import pandas as pd

    if label not in _OPP_LABEL_COL:
        raise ValueError(f"label must be one of {list(_OPP_LABEL_COL)}")

    folder = os.path.join(data_dir, "opportunity")
    files = sorted(glob.glob(os.path.join(folder, "**", "S*.dat"), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No OPPORTUNITY .dat files found under '{folder}'. Unpack the UCI "
            f"archive there (or run download_datasets.py)."
        )

    label_col = _OPP_LABEL_COL[label]
    X_parts, y_parts = [], []
    for fp in files:
        arr = pd.read_csv(fp, sep=r"\s+", header=None, na_values=["NaN"]).values
        if row_stride > 1:
            arr = arr[::row_stride]
        X_parts.append(arr[:, _OPP_SENSOR_SLICE])
        y_parts.append(arr[:, label_col])

    X = np.vstack(X_parts).astype(float)
    y = np.concatenate(y_parts).astype(float)

    # Drop the Null / unlabelled rows.
    if drop_null:
        keep = y != 0
        X, y = X[keep], y[keep]

    # Drop sensor channels that are entirely missing (cannot be imputed).
    valid_cols = ~np.all(np.isnan(X), axis=0)
    X = X[:, valid_cols]

    # Stratified subsample to keep the kernel SVM tractable.
    if max_samples is not None and X.shape[0] > max_samples:
        rng = np.random.RandomState(random_state)
        idx = []
        classes, counts = np.unique(y, return_counts=True)
        for c, n in zip(classes, counts):
            c_idx = np.where(y == c)[0]
            take = max(1, int(round(max_samples * n / len(y))))
            idx.append(rng.choice(c_idx, size=min(take, len(c_idx)), replace=False))
        idx = np.concatenate(idx)
        rng.shuffle(idx)
        X, y = X[idx], y[idx]

    # Integer-encode labels to 0..K-1.
    classes = np.unique(y)
    remap = {c: i for i, c in enumerate(classes)}
    y = np.array([remap[v] for v in y], dtype=int)

    return Dataset(X=X, y=y, name=f"opportunity[{label}]")


def _casas_parse_label(field: str):
    """Return (activity_name, marker) where marker in {'begin','end',None}.

    CASAS label fields look like: Toilet, Toilet="begin", Toilet="end".
    Empty / Other_Activity -> (None, None).
    """
    field = (field or "").strip()
    if not field or field.lower() in ("other_activity", "other", "none"):
        return None, None
    marker = None
    if '="begin"' in field or field.endswith('=begin'):
        marker = "begin"
    elif '="end"' in field or field.endswith('=end'):
        marker = "end"
    name = field.split("=")[0].strip().strip('"')
    if not name or name.lower().startswith("other"):
        return None, None
    return name, marker


def load_casas(
    data_dir: str = DEFAULT_DATA_DIR,
    home: str = "hh101.csv",
    window_events: int = 30,
    step: int = 15,
    min_activity_count: int = 40,
    top_k_activities: Optional[int] = 12,
    max_samples: Optional[int] = 5000,
    random_state: int = 0,
) -> Dataset:
    """Loader for the CASAS smart-home dataset (ambient sensor event logs).

    Reference
    ---------
    Cook et al. (2013), "CASAS: A Smart Home in a Box", IEEE Computer.

    Expects the unpacked record under ``data_dir/casas/`` with the labelled
    event logs in ``data_dir/casas/labeled/`` (one CSV per home, e.g.
    ``hh101.csv``). Each line is:
        DATE,TIME,SENSOR_ID,MESSAGE[,ACTIVITY_LABEL]
    where ambient sensors include motion (M), door (D), area (A), light (L),
    and temperature (T). Activity labels are sparse and delimited by
    ``="begin"`` / ``="end"`` markers.

    Feature extraction (event windowing)
    ------------------------------------
    A single home is used so the sensor set is consistent. Events are scanned
    while tracking the current activity (forward-filled between begin/end
    markers). The event stream is cut into overlapping windows of
    ``window_events`` events (stride ``step``). For each window the feature
    vector is the per-sensor activation count, plus the (cyclically encoded)
    hour of day. The window label is the dominant activity among its events;
    windows with no labelled activity are dropped.

    Parameters
    ----------
    home : str
        CSV file name in ``labeled/`` to use (default "hh101.csv").
    window_events, step : int
        Window length and stride, in number of events.
    min_activity_count : int
        Drop activity classes with fewer than this many windows.
    max_samples : int or None
        Stratified subsample cap to keep the kernel SVM tractable.
    random_state : int
        Seed for the subsample.

    Returns
    -------
    Dataset
    """
    import pandas as pd

    folder = os.path.join(data_dir, "casas")
    # the labelled logs live in casas/labeled/ ; fall back to casas/ itself
    cand = [os.path.join(folder, "labeled", home), os.path.join(folder, home)]
    path = next((p for p in cand if os.path.isfile(p)), None)
    if path is None:
        raise FileNotFoundError(
            f"CASAS home '{home}' not found under '{folder}'. Unpack the Zenodo "
            f"record there (labelled logs go in casas/labeled/), or run "
            f"download_datasets.py --only casas."
        )

    df = pd.read_csv(
        path, header=None, names=["date", "time", "sensor", "message", "label"],
        dtype=str, keep_default_na=False, on_bad_lines="skip", engine="python",
    )

    sensors = df["sensor"].values
    times = df["time"].values
    labels = df["label"].values

    # Forward-fill the current activity through the event stream.
    current = None
    event_acts = []
    for lab in labels:
        name, marker = _casas_parse_label(lab)
        if marker == "begin":
            current = name
            event_acts.append(name)
        elif marker == "end":
            event_acts.append(name if name else current)
            current = None
        elif name is not None:
            current = name
            event_acts.append(name)
        else:
            event_acts.append(current)

    # Sensor vocabulary -> feature index.
    sensor_vocab = sorted(set(sensors))
    sidx = {s: i for i, s in enumerate(sensor_vocab)}
    n_feat = len(sensor_vocab) + 2  # + sin/cos hour

    def hour_of(t: str) -> float:
        try:
            return int(t.split(":")[0])
        except Exception:
            return 0

    X_rows, y_rows = [], []
    n = len(sensors)
    for start in range(0, n - window_events + 1, step):
        end = start + window_events
        vec = np.zeros(n_feat, dtype=float)
        for j in range(start, end):
            vec[sidx[sensors[j]]] += 1.0
        h = hour_of(times[start])
        vec[-2] = np.sin(2 * np.pi * h / 24.0)
        vec[-1] = np.cos(2 * np.pi * h / 24.0)
        # dominant labelled activity in the window
        win_acts = [a for a in event_acts[start:end] if a]
        if not win_acts:
            continue
        vals, counts = np.unique(np.array(win_acts), return_counts=True)
        y_rows.append(vals[int(np.argmax(counts))])
        X_rows.append(vec)

    if not X_rows:
        raise RuntimeError(f"No labelled windows extracted from {path}.")

    X = np.vstack(X_rows)
    y = np.array(y_rows)

    # Drop rare activity classes, then keep only the top-K most frequent.
    classes, counts = np.unique(y, return_counts=True)
    keep_classes = set(classes[counts >= min_activity_count])
    if top_k_activities is not None:
        order = np.argsort(counts)[::-1]
        top = set(classes[order][:top_k_activities])
        keep_classes &= top
    mask = np.array([v in keep_classes for v in y])
    X, y = X[mask], y[mask]

    # Stratified subsample.
    if max_samples is not None and X.shape[0] > max_samples:
        rng = np.random.RandomState(random_state)
        idx = []
        classes, counts = np.unique(y, return_counts=True)
        for c, cnt in zip(classes, counts):
            c_idx = np.where(y == c)[0]
            take = max(1, int(round(max_samples * cnt / len(y))))
            idx.append(rng.choice(c_idx, size=min(take, len(c_idx)), replace=False))
        idx = np.concatenate(idx)
        rng.shuffle(idx)
        X, y = X[idx], y[idx]

    # Integer-encode labels.
    classes = np.unique(y)
    remap = {c: i for i, c in enumerate(classes)}
    y = np.array([remap[v] for v in y], dtype=int)

    return Dataset(X=X, y=y, name=f"casas[{home}]")


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
