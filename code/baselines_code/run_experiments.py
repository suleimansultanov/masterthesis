"""
run_experiments.py
==================
Main entry point. Runs the full baseline experimental protocol on the synthetic
IoT-healthcare dataset (out of the box) or on a real dataset if available.

Usage
-----
    python run_experiments.py                  # synthetic, default small config
    python run_experiments.py --dataset opportunity
    python run_experiments.py --n-samples 600 --n-features 80 --n-repeats 5

Outputs (written to ./results/):
    summary_results.csv     - mean +/- std accuracy & macro-F1 per model
    noise_curves.csv        - accuracy & F1 vs sigma per model
    wilcoxon_results.csv    - paired significance tests (proposed vs baselines)
    noise_curves.png        - degradation plot (if matplotlib available)
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from data import load_dataset
from evaluation import ExperimentHarness
from models import (
    AdaptiveHybridKernelSVM,
    SVMLinear,
    SVMRBF,
    StaticMKL,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def parse_args():
    p = argparse.ArgumentParser(description="Adaptive Hybrid Kernel SVM baselines")
    p.add_argument("--dataset", default="synthetic",
                   choices=["synthetic", "opportunity", "casas"])
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--n-features", type=int, default=60)
    p.add_argument("--n-classes", type=int, default=4)
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument("--n-cv-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-proposed", action="store_true",
                   help="Also try the proposed model stub (will be skipped "
                        "until its fit() is implemented).")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Load data ----
    if args.dataset == "synthetic":
        ds = load_dataset(
            "synthetic",
            n_samples=args.n_samples,
            n_features=args.n_features,
            n_classes=args.n_classes,
            random_state=args.seed,
        )
    else:
        ds = load_dataset(args.dataset)
    print(ds.summary())

    # ---- Register models ----
    models = {
        SVMRBF.name: SVMRBF,
        SVMLinear.name: SVMLinear,
        StaticMKL.name: StaticMKL,
    }
    if args.include_proposed:
        models[AdaptiveHybridKernelSVM.name] = AdaptiveHybridKernelSVM

    harness = ExperimentHarness(
        models=models,
        n_repeats=args.n_repeats,
        n_cv_splits=args.n_cv_splits,
        base_seed=args.seed,
        noise_sigmas=[0.1, 0.3, 0.5, 0.7, 1.0],
        noise_fraction=0.30,
    )

    harness.run(ds.X, ds.y)

    # ---- Summary table ----
    summary = harness.summary_table()
    print("\n================ SUMMARY (mean +/- std over repeats) ================")
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(summary.to_string(index=False))
    summary.to_csv(os.path.join(RESULTS_DIR, "summary_results.csv"), index=False)

    # ---- Noise curves ----
    noise = harness.noise_curve_table()
    print("\n================ NOISE-ROBUSTNESS CURVES ================")
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(noise.to_string(index=False))
    noise.to_csv(os.path.join(RESULTS_DIR, "noise_curves.csv"), index=False)

    # ---- Wilcoxon (proposed vs baselines) ----
    wil = harness.wilcoxon_table(AdaptiveHybridKernelSVM.name)
    print("\n================ WILCOXON SIGNED-RANK (proposed vs baselines) ======")
    with pd.option_context("display.width", 140, "display.max_columns", None):
        print(wil.to_string(index=False))
    wil.to_csv(os.path.join(RESULTS_DIR, "wilcoxon_results.csv"), index=False)

    # ---- Plot (optional) ----
    _maybe_plot_noise_curves(noise)

    print(f"\nAll result CSVs written to: {RESULTS_DIR}")


def _maybe_plot_noise_curves(noise_df: pd.DataFrame):
    if noise_df.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib not available; skipping PNG plot)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for metric, ax in zip(["accuracy_mean", "f1_macro_mean"], axes):
        for model, g in noise_df.groupby("model"):
            g = g.sort_values("sigma")
            std_col = metric.replace("_mean", "_std")
            ax.errorbar(g["sigma"], g[metric], yerr=g[std_col],
                        marker="o", capsize=3, label=model)
        ax.set_xlabel("Noise sigma (std on 30% of features, test-time)")
        ax.set_ylabel(metric.replace("_mean", "").replace("_", " ").title())
        ax.set_title(f"Degradation: {metric.replace('_mean','')}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "noise_curves.png")
    fig.savefig(out, dpi=120)
    print(f"(saved noise-curve plot to {out})")


if __name__ == "__main__":
    main()
