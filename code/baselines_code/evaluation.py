"""
evaluation.py
=============
Experimental harness implementing the thesis protocol:

  - 5-fold STRATIFIED CV for hyperparameter tuning on the train split.
  - The whole experiment (train/test split + CV + test) repeats N times with
    different seeds. Results reported as mean +/- std.
  - Primary metrics: Accuracy and Macro-averaged F1.
  - Noise-robustness test: inject Gaussian noise (sigma grid) into a random 30%
    of feature dimensions at TEST time only -> degradation curves.
  - Statistical significance: paired Wilcoxon signed-rank test across the
    repeats x folds fold-level results, proposed model vs each baseline.

The harness is model-agnostic: any object implementing the BaseModel interface
(fit/predict/set_params/param_grid) can be registered and evaluated. Models
whose fit() raises NotImplementedError (e.g. the proposed-model stub) are
skipped gracefully so the baseline pipeline still runs end to end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from preprocessing import Preprocessor, inject_gaussian_noise


# ---------------------------------------------------------------------------
# CV hyperparameter tuning
# ---------------------------------------------------------------------------
def tune_hyperparameters(
    model_cls,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
    scoring: str = "f1_macro",
    random_state: int = 0,
):
    """Grid-search a model's param_grid with stratified k-fold CV.

    Returns the best params dict and the best mean CV score.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_params, best_score = None, -np.inf

    for params in model_cls.param_grid():
        fold_scores = []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            model = model_cls().set_params(**params)
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)
            if scoring == "accuracy":
                fold_scores.append(accuracy_score(y_va, pred))
            else:  # f1_macro
                fold_scores.append(
                    f1_score(y_va, pred, average="macro", zero_division=0)
                )
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score, best_params = mean_score, params

    return best_params, best_score


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class ModelResults:
    name: str
    accuracies: List[float] = field(default_factory=list)
    f1s: List[float] = field(default_factory=list)
    # fold-level scores stacked across repeats, used for Wilcoxon
    fold_accuracies: List[float] = field(default_factory=list)
    fold_f1s: List[float] = field(default_factory=list)
    # noise curves: sigma -> list of (acc, f1) across repeats
    noise_acc: Dict[float, List[float]] = field(default_factory=dict)
    noise_f1: Dict[float, List[float]] = field(default_factory=dict)
    best_params_per_repeat: List[dict] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str = ""

    def summary_row(self) -> dict:
        return {
            "model": self.name,
            "accuracy_mean": np.mean(self.accuracies) if self.accuracies else np.nan,
            "accuracy_std": np.std(self.accuracies) if self.accuracies else np.nan,
            "f1_macro_mean": np.mean(self.f1s) if self.f1s else np.nan,
            "f1_macro_std": np.std(self.f1s) if self.f1s else np.nan,
            "skipped": self.skipped,
        }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------
class ExperimentHarness:
    def __init__(
        self,
        models: Dict[str, Type],
        n_repeats: int = 5,
        n_cv_splits: int = 5,
        test_size: float = 0.3,
        noise_sigmas: Optional[List[float]] = None,
        noise_fraction: float = 0.30,
        base_seed: int = 42,
        scoring: str = "f1_macro",
    ):
        self.models = models
        self.n_repeats = n_repeats
        self.n_cv_splits = n_cv_splits
        self.test_size = test_size
        self.noise_sigmas = noise_sigmas or [0.1, 0.3, 0.5, 0.7, 1.0]
        self.noise_fraction = noise_fraction
        self.base_seed = base_seed
        self.scoring = scoring
        self.results: Dict[str, ModelResults] = {
            name: ModelResults(name=name) for name in models
        }

    def run(self, X: np.ndarray, y: np.ndarray):
        """Run repeated train/test + CV + test + noise-robustness for all models."""
        for name in self.results:
            self.results[name].noise_acc = {s: [] for s in self.noise_sigmas}
            self.results[name].noise_f1 = {s: [] for s in self.noise_sigmas}

        for r in range(self.n_repeats):
            seed = self.base_seed + r
            print(f"\n=== Repeat {r + 1}/{self.n_repeats} (seed={seed}) ===")

            X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
                X, y, test_size=self.test_size, stratify=y, random_state=seed
            )

            # Preprocess: fit imputer + scaler on TRAIN only, apply to both.
            pre = Preprocessor(impute_strategy="mean").fit(X_tr_raw)
            X_tr = pre.transform(X_tr_raw)
            X_te = pre.transform(X_te_raw)

            for name, model_cls in self.models.items():
                res = self.results[name]
                try:
                    best_params, cv_score = tune_hyperparameters(
                        model_cls,
                        X_tr,
                        y_tr,
                        n_splits=self.n_cv_splits,
                        scoring=self.scoring,
                        random_state=seed,
                    )
                    model = model_cls().set_params(**best_params)
                    model.fit(X_tr, y_tr)
                    pred = model.predict(X_te)
                    acc = accuracy_score(y_te, pred)
                    f1 = f1_score(y_te, pred, average="macro", zero_division=0)

                    res.accuracies.append(acc)
                    res.f1s.append(f1)
                    res.best_params_per_repeat.append(best_params)

                    # Fold-level CV scores (for Wilcoxon across repeats x folds).
                    fold_acc, fold_f1 = self._cv_fold_scores(
                        model_cls, best_params, X_tr, y_tr, seed
                    )
                    res.fold_accuracies.extend(fold_acc)
                    res.fold_f1s.extend(fold_f1)

                    # Noise-robustness curves (test-time noise only).
                    for sigma in self.noise_sigmas:
                        X_te_noisy, _ = inject_gaussian_noise(
                            X_te,
                            sigma=sigma,
                            fraction=self.noise_fraction,
                            random_state=seed,
                        )
                        pred_n = model.predict(X_te_noisy)
                        res.noise_acc[sigma].append(accuracy_score(y_te, pred_n))
                        res.noise_f1[sigma].append(
                            f1_score(y_te, pred_n, average="macro", zero_division=0)
                        )

                    print(
                        f"  [{name:28s}] acc={acc:.4f}  f1={f1:.4f}  "
                        f"(cv {self.scoring}={cv_score:.4f}, params={best_params})"
                    )

                except NotImplementedError as e:
                    res.skipped = True
                    res.skip_reason = str(e).split("\n")[0]
                    print(f"  [{name:28s}] SKIPPED (not implemented)")
                except Exception as e:  # keep the harness robust
                    res.skipped = True
                    res.skip_reason = repr(e)
                    print(f"  [{name:28s}] ERROR: {e!r}")

        return self.results

    def _cv_fold_scores(self, model_cls, params, X_tr, y_tr, seed):
        """Re-evaluate the chosen params per fold to collect fold-level scores
        used for the paired Wilcoxon test."""
        skf = StratifiedKFold(
            n_splits=self.n_cv_splits, shuffle=True, random_state=seed
        )
        accs, f1s = [], []
        for tr_idx, va_idx in skf.split(X_tr, y_tr):
            m = model_cls().set_params(**params)
            m.fit(X_tr[tr_idx], y_tr[tr_idx])
            p = m.predict(X_tr[va_idx])
            accs.append(accuracy_score(y_tr[va_idx], p))
            f1s.append(f1_score(y_tr[va_idx], p, average="macro", zero_division=0))
        return accs, f1s

    # ----- Reporting -------------------------------------------------------
    def summary_table(self) -> pd.DataFrame:
        rows = [self.results[n].summary_row() for n in self.results]
        return pd.DataFrame(rows)

    def noise_curve_table(self) -> pd.DataFrame:
        rows = []
        for name, res in self.results.items():
            if res.skipped:
                continue
            for sigma in self.noise_sigmas:
                accs = res.noise_acc.get(sigma, [])
                f1s = res.noise_f1.get(sigma, [])
                if not accs:
                    continue
                rows.append(
                    {
                        "model": name,
                        "sigma": sigma,
                        "accuracy_mean": np.mean(accs),
                        "accuracy_std": np.std(accs),
                        "f1_macro_mean": np.mean(f1s),
                        "f1_macro_std": np.std(f1s),
                    }
                )
        return pd.DataFrame(rows)

    def wilcoxon_table(self, proposed_name: str) -> pd.DataFrame:
        """Paired Wilcoxon signed-rank test comparing the proposed model to
        each baseline across the repeats x folds fold-level scores."""
        rows = []
        prop = self.results.get(proposed_name)
        if prop is None or prop.skipped or not prop.fold_f1s:
            # The proposed model is a stub by default -> nothing to compare.
            return pd.DataFrame(
                [
                    {
                        "comparison": f"{proposed_name} vs (n/a)",
                        "note": "proposed model not implemented; "
                        "no Wilcoxon test performed",
                    }
                ]
            )
        for name, res in self.results.items():
            if name == proposed_name or res.skipped:
                continue
            n = min(len(prop.fold_f1s), len(res.fold_f1s))
            a = np.array(prop.fold_f1s[:n])
            b = np.array(res.fold_f1s[:n])
            try:
                stat, p = wilcoxon(a, b)
            except ValueError:
                stat, p = np.nan, np.nan
            rows.append(
                {
                    "comparison": f"{proposed_name} vs {name}",
                    "metric": "f1_macro (fold-level)",
                    "n_pairs": n,
                    "median_diff": float(np.median(a - b)),
                    "wilcoxon_stat": stat,
                    "p_value": p,
                    "significant_p<0.05": (p < 0.05) if p == p else False,
                }
            )
        return pd.DataFrame(rows)
