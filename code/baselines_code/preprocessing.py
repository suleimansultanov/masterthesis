"""
preprocessing.py
================
Preprocessing utilities:
  - Missing-value / sensor-dropout imputation.
  - Standardization (zero mean, unit variance) fitted on TRAIN only.
  - Test-time Gaussian noise injection into a random subset of feature
    dimensions (for the noise-robustness experiment).

The standardizer is implemented as a small wrapper around sklearn so the
imputation + scaling can be fit on the training fold and re-applied to the
test fold (no leakage).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """Impute missing values then standardize.

    Fit on train, transform train and test. This prevents information leakage
    from the test fold into the scaling statistics.
    """

    def __init__(self, impute_strategy: str = "mean"):
        self.imputer = SimpleImputer(strategy=impute_strategy)
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray) -> "Preprocessor":
        X_imp = self.imputer.fit_transform(X)
        self.scaler.fit(X_imp)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit before transform.")
        X_imp = self.imputer.transform(X)
        return self.scaler.transform(X_imp)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------
# Test-time noise injection (robustness experiment)
# ---------------------------------------------------------------------------
def inject_gaussian_noise(
    X: np.ndarray,
    sigma: float,
    fraction: float = 0.30,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject zero-mean Gaussian noise with std ``sigma`` into a random subset
    of feature dimensions.

    Per the thesis protocol this is applied at TEST time only, to a random 30%
    of feature dimensions by default. Because the data is standardized before
    this step, ``sigma`` is expressed in units of feature standard deviations.

    Parameters
    ----------
    X : ndarray [n_samples, n_features]
        (Already standardized) test features.
    sigma : float
        Standard deviation of the injected Gaussian noise.
    fraction : float
        Fraction of feature dimensions to corrupt (default 0.30).
    random_state : int or None
        Seed controlling both which dimensions are chosen and the noise draw.

    Returns
    -------
    X_noisy : ndarray
        Copy of X with noise added to the selected columns.
    noisy_cols : ndarray
        Indices of the corrupted columns (useful for logging / debugging).
    """
    rng = np.random.RandomState(random_state)
    X_noisy = X.copy()
    n_features = X.shape[1]
    n_noisy = max(1, int(round(fraction * n_features)))
    noisy_cols = rng.choice(n_features, size=n_noisy, replace=False)

    if sigma > 0:
        noise = rng.normal(0.0, sigma, size=(X.shape[0], n_noisy))
        X_noisy[:, noisy_cols] += noise

    return X_noisy, noisy_cols


def simulate_dropout(
    X: np.ndarray,
    dropout_rate: float,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Randomly set a fraction of entries to NaN to simulate sensor dropout.

    Useful for additional robustness experiments. The NaNs can subsequently be
    handled by a Preprocessor / imputer.
    """
    rng = np.random.RandomState(random_state)
    X_out = X.copy()
    mask = rng.rand(*X.shape) < dropout_rate
    X_out[mask] = np.nan
    return X_out
