"""
models.py
=========
Models for the thesis. All models share a common scikit-learn-style
fit / predict interface and expose a hyperparameter grid for CV tuning, so the
evaluation harness can treat them uniformly.

Implemented:
  1. SVMRBF     - standard SVM, RBF kernel; tune (C, gamma); no feature weights.
  2. SVMLinear  - standard SVM, linear kernel; tune C; no feature weights.
  3. StaticMKL  - convex combination K = mu*K_linear + (1-mu)*K_rbf via a
                  precomputed Gram matrix; tune (C, mu, gamma); NO feature
                  weights. Key ablation baseline.

Stub (proposed model):
  4. AdaptiveHybridKernelSVM - same interface; the dynamic per-feature
     weighting logic is left as a clearly-marked TODO / NotImplemented
     placeholder so it plugs into the same harness later.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
from sklearn.svm import SVC

from kernels import combined_kernel


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------
class BaseModel(ABC):
    """Common interface for all baselines and the proposed model."""

    name: str = "base"

    @abstractmethod
    def set_params(self, **params) -> "BaseModel":
        """Set hyperparameters (used during CV tuning)."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    @abstractmethod
    def param_grid() -> List[Dict]:
        """Return a list of hyperparameter dicts to search over in CV."""

    def clone(self) -> "BaseModel":
        """Return a fresh, unfitted instance of the same class."""
        return self.__class__()


def _expand_grid(grid: Dict[str, list]) -> List[Dict]:
    """Cartesian product of a dict-of-lists into a list-of-dicts."""
    keys = list(grid.keys())
    combos = itertools.product(*[grid[k] for k in keys])
    return [dict(zip(keys, c)) for c in combos]


# ---------------------------------------------------------------------------
# 1. SVM with RBF kernel
# ---------------------------------------------------------------------------
class SVMRBF(BaseModel):
    name = "SVM-RBF"

    def __init__(self, C: float = 1.0, gamma: float = "scale"):
        self.C = C
        self.gamma = gamma
        self._clf = None

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        # class_weight='balanced' helps with the imbalanced multi-class setting.
        self._clf = SVC(
            kernel="rbf", C=self.C, gamma=self.gamma, class_weight="balanced"
        )
        self._clf.fit(X, y)
        return self

    def predict(self, X):
        return self._clf.predict(X)

    @staticmethod
    def param_grid():
        return _expand_grid(
            {
                "C": [0.1, 1.0, 10.0, 100.0],
                "gamma": ["scale", 0.01, 0.1, 1.0],
            }
        )


# ---------------------------------------------------------------------------
# 2. SVM with linear kernel
# ---------------------------------------------------------------------------
class SVMLinear(BaseModel):
    name = "SVM-Linear"

    def __init__(self, C: float = 1.0):
        self.C = C
        self._clf = None

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        self._clf = SVC(kernel="linear", C=self.C, class_weight="balanced")
        self._clf.fit(X, y)
        return self

    def predict(self, X):
        return self._clf.predict(X)

    @staticmethod
    def param_grid():
        return _expand_grid({"C": [0.01, 0.1, 1.0, 10.0, 100.0]})


# ---------------------------------------------------------------------------
# 3. Static MKL (precomputed combined Gram matrix) -- key ablation baseline
# ---------------------------------------------------------------------------
class StaticMKL(BaseModel):
    """Static Multiple Kernel Learning: a fixed convex combination of a linear
    and an RBF kernel, K = mu*K_linear + (1-mu)*K_rbf, with mu tuned by CV.

    Implemented with SVC(kernel='precomputed'). NO feature weights -- this is
    precisely the hybrid kernel WITHOUT the adaptive feature weighting, i.e.
    the ablation against which the proposed model is compared.
    """

    name = "Static-MKL"

    def __init__(self, C: float = 1.0, mu: float = 0.5, gamma: float = 1.0):
        self.C = C
        self.mu = mu
        self.gamma = gamma
        self._clf = None
        self._X_train = None  # stored to build the test-time Gram matrix

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        self._X_train = np.asarray(X, dtype=float)
        K_train = combined_kernel(
            self._X_train, mu=self.mu, gamma=self.gamma, feature_weights=None
        )
        self._clf = SVC(kernel="precomputed", C=self.C, class_weight="balanced")
        self._clf.fit(K_train, y)
        return self

    def predict(self, X):
        if self._clf is None:
            raise RuntimeError("Model must be fit before predict.")
        # Gram matrix between test points (rows) and train points (cols).
        K_test = combined_kernel(
            np.asarray(X, dtype=float),
            self._X_train,
            mu=self.mu,
            gamma=self.gamma,
            feature_weights=None,
        )
        return self._clf.predict(K_test)

    @staticmethod
    def param_grid():
        return _expand_grid(
            {
                "C": [1.0, 10.0, 100.0],
                "mu": [0.25, 0.5, 0.75],
                "gamma": [0.01, 0.1, 1.0],
            }
        )


# ---------------------------------------------------------------------------
# 4. Proposed model -- Adaptive Hybrid Kernel SVM (STUB)
# ---------------------------------------------------------------------------
class AdaptiveHybridKernelSVM(BaseModel):
    """Proposed model: a hybrid linear+RBF kernel SVM with DYNAMIC per-feature
    weighting for noisy, high-dimensional IoT-healthcare data.

    The architecture deliberately reuses ``combined_kernel`` with a
    ``feature_weights`` vector, so once the weighting logic is implemented the
    model drops into the same evaluation harness as the baselines.

    STATUS: stub. The fit() body that LEARNS the adaptive feature weights is a
    TODO. Everything else (interface, param grid, prediction path given fixed
    weights) is wired up.
    """

    name = "Adaptive-Hybrid-Kernel-SVM"

    def __init__(self, C: float = 1.0, mu: float = 0.5, gamma: float = 1.0):
        self.C = C
        self.mu = mu
        self.gamma = gamma
        self.feature_weights_ = None  # learned per-feature weights (TODO)
        self._clf = None
        self._X_train = None

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y):
        # ------------------------------------------------------------------
        # TODO (CORE THESIS CONTRIBUTION -- NOT YET IMPLEMENTED):
        #   Learn a vector of per-feature weights w in R^{n_features} that
        #   down-weights noisy / uninformative sensor dimensions and
        #   up-weights discriminative ones, e.g. by:
        #     - alternating optimization between SVM dual and feature weights,
        #     - gradient descent on a kernel-target-alignment objective, or
        #     - a margin/relevance criterion estimated per fold.
        #   The learned weights then feed combined_kernel(..., feature_weights=w).
        # ------------------------------------------------------------------
        raise NotImplementedError(
            "AdaptiveHybridKernelSVM.fit: the dynamic feature-weighting logic "
            "is the core thesis contribution and is intentionally left as a "
            "TODO. Implement weight learning here, store it in "
            "self.feature_weights_, then build the kernel with combined_kernel("
            "..., feature_weights=self.feature_weights_)."
        )

    def predict(self, X):
        # Once fit() is implemented, this path already works:
        if self._clf is None:
            raise RuntimeError("Model must be fit before predict.")
        K_test = combined_kernel(
            np.asarray(X, dtype=float),
            self._X_train,
            mu=self.mu,
            gamma=self.gamma,
            feature_weights=self.feature_weights_,
        )
        return self._clf.predict(K_test)

    @staticmethod
    def param_grid():
        return _expand_grid(
            {
                "C": [1.0, 10.0, 100.0],
                "mu": [0.25, 0.5, 0.75],
                "gamma": [0.01, 0.1, 1.0],
            }
        )


# Registry of available models for the harness.
MODEL_REGISTRY = {
    SVMRBF.name: SVMRBF,
    SVMLinear.name: SVMLinear,
    StaticMKL.name: StaticMKL,
    AdaptiveHybridKernelSVM.name: AdaptiveHybridKernelSVM,
}
