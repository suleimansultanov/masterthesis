---
title: Adaptive Hybrid Kernel SVM
type: concept
tags: [concept, core-contribution, kernel, svm]
---

# Adaptive Hybrid Kernel SVM

> [!success] Core contribution тезиса
> Композитное ядро из двух комплементарных компонентов + data-quality-driven [[Feature Weighting|веса]].

Rationale: **linear** компонент ловит глобальную cross-feature структуру; **RBF** — fine-grained нелинейные взаимодействия и локальную similarity. Веса гарантируют, что ненадёжные фичи меньше вносят в оба компонента → сама similarity-мера робастнее.

$$ K(x_i, x_j) = \mu\, K_{lin}(W\odot x_i, W\odot x_j) + (1-\mu)\, K_{rbf}(W\odot x_i, W\odot x_j) $$

- $\mu \in [0,1]$ — баланс global/local (adaptive mixture coefficient).
- $W$ — per-feature reliability веса, применяются **до** ядра.

Объединяет линии [[Multiple Kernel Learning]] и [[Feature Weighting]], раньше развивавшиеся раздельно → см. [[Research Gap and Novelty]].

Полная математика, оптимизация и псевдокод → [[Mathematical Formulation]].
