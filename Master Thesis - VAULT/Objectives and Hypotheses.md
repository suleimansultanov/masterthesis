---
title: Objectives and Hypotheses
type: note
tags: [thesis, objectives, hypotheses]
---

# Objectives and Hypotheses

## Primary objective
Спроектировать и оценить [[Adaptive Hybrid Kernel SVM]] — global linear + local RBF с (a) adaptive mixture coefficient $\mu$ и (b) data-quality-driven per-feature весами — для классификации шумных высокоразмерных IoT healthcare данных. Цель: измеримый прирост над single-kernel SVM и static MKL по accuracy, macro-F1 и noise robustness.

## Sub-objectives
1. **Mathematical formulation** — композитное ядро, вычисление весов, SVM-objective, hyperparameter space → [[Mathematical Formulation]].
2. **Implementation** — расширение стандартных SVM-солверов в Python/scikit-learn через custom precomputed kernel matrix.
3. **Empirical validation** — на [[Datasets|OPPORTUNITY + CASAS]] против трёх [[Baselines]].
4. **Noise robustness analysis** — synthetic noise injection при разных интенсивностях.

## Hypotheses
> [!question] H1 — Hybrid kernel advantage
> На мультимодальных шумных датасетах adaptive hybrid kernel SVM даст выше accuracy и macro-F1, чем любое одиночное фиксированное ядро (RBF или linear).

> [!question] H2 — Noise resilience
> Data-quality-driven веса в ядре дадут лучшую устойчивость к synthetic noise injection, чем стандартный kernel SVM или static MKL без весов → меньшая деградация при росте интенсивности шума.

> [!question] H3 — Feature weighting contribution
> Механизм весов даст измеримое улучшение **над hybrid kernel без весов** (тот же linear-RBF mix). Тестируется ablation'ом полной модели против варианта без весов → см. [[Experimental Protocol]].
