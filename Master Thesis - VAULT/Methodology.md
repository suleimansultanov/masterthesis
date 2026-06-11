---
title: Methodology
type: note
tags: [thesis, methodology]
---

# Methodology

Цель — воспроизводимость и feasibility в рамках Master's.

## Пайплайн
1. [[Mathematical Formulation]] — модель, веса, оптимизация, псевдокод.
2. Data handling → [[Datasets]] + препроцессинг (ниже).
3. [[Experimental Protocol]] — CV, метрики, noise test, ablation, significance.
4. Implementation details (стек).

## Data preprocessing (одинаковый для обоих датасетов)
1. **Missing value handling** — forward-fill (LOCF), затем mean imputation для остатка. Missingness каждого признака фиксируется **до** импутации для [[Feature Weighting|весов]].
2. **Robust normalization** — robust scaling (median + IQR), снижает влияние выбросов.
3. **Feature extraction** — OPPORTUNITY: статфичи (mean, std, min, max, energy) по sliding windows; CASAS: sensor event counts + temporal features per activity window.
4. **Train-test split** — 80/20, стратифицировано по классу. CV только на train, test зарезервирован для финала.

## Implementation stack
- **scikit-learn (v1.3+)** — SVC с precomputed kernel matrix (кастомное ядро без правки ядра оптимизатора).
- **NumPy/SciPy** — численные вычисления, построение kernel matrix, стат-тесты.
- **Pandas** — загрузка, препроцессинг, feature extraction.
- **Matplotlib/Seaborn** — визуализация (noise degradation curves, распределения весов).

> [!note] Hardware
> Обычный workstation, без GPU (SVM на этих размерах не выигрывает от GPU). Код, конфиги, random seeds и параметры препроцессинга документируются; полный codebase сдаётся с тезисом.
