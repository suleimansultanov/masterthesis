"""
kernels.py
==========
Kernel / Gram-matrix utilities.

Provides:
  - linear_kernel(X, Y)            -> linear Gram matrix
  - rbf_kernel_matrix(X, Y, gamma) -> RBF Gram matrix
  - combined_kernel(...)           -> Static MKL convex combination
                                      K = mu * K_linear + (1 - mu) * K_rbf

These are used by the precomputed-kernel SVM baselines (notably Static MKL)
and are also the natural place where the proposed Adaptive Hybrid Kernel model
will later apply per-feature weights.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import linear_kernel as _sk_linear
from sklearn.metrics.pairwise import rbf_kernel as _sk_rbf


def linear_kernel(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Linear Gram matrix K[i, j] = <x_i, y_j>."""
    if Y is None:
        Y = X
    return _sk_linear(X, Y)


def rbf_kernel_matrix(
    X: np.ndarray, Y: np.ndarray | None = None, gamma: float = 1.0
) -> np.ndarray:
    """RBF Gram matrix K[i, j] = exp(-gamma * ||x_i - y_j||^2)."""
    if Y is None:
        Y = X
    return _sk_rbf(X, Y, gamma=gamma)


def combined_kernel(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    mu: float = 0.5,
    gamma: float = 1.0,
    feature_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Static MKL convex combination of linear and RBF kernels.

        K = mu * K_linear + (1 - mu) * K_rbf

    Parameters
    ----------
    X : ndarray [n_samples_X, n_features]
    Y : ndarray [n_samples_Y, n_features] or None
        If None, Y = X (training Gram matrix).
    mu : float in [0, 1]
        Mixing coefficient. mu=1 -> pure linear, mu=0 -> pure RBF.
    gamma : float
        RBF bandwidth.
    feature_weights : ndarray [n_features] or None
        Optional per-feature weights applied as a diagonal scaling before the
        kernels are computed (X' = X * sqrt(w)). The Static MKL baseline passes
        None (no feature weighting); this hook exists so the proposed Adaptive
        Hybrid Kernel model can reuse the same code path.

    Returns
    -------
    K : ndarray [n_samples_X, n_samples_Y]
    """
    if not 0.0 <= mu <= 1.0:
        raise ValueError(f"mu must be in [0, 1], got {mu}")
    if Y is None:
        Y = X

    if feature_weights is not None:
        w = np.asarray(feature_weights, dtype=float)
        if w.shape[0] != X.shape[1]:
            raise ValueError("feature_weights length must equal n_features")
        sqrt_w = np.sqrt(np.clip(w, 0.0, None))
        X = X * sqrt_w
        Y = Y * sqrt_w

    K_lin = linear_kernel(X, Y)
    K_rbf = rbf_kernel_matrix(X, Y, gamma=gamma)
    return mu * K_lin + (1.0 - mu) * K_rbf
