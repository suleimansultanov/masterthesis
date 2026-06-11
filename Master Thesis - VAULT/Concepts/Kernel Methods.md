---
title: Kernel Methods
type: concept
tags: [concept, kernel, svm]
---

# Kernel Methods

Kernel trick неявно отображает входы в более высокоразмерное пространство, где линейное разделение становится возможным.

## Базовые ядра
- **Linear**: $K(x,y) = x^\top y$ — для приблизительно линейно разделимых данных, глобальные тренды.
- **Polynomial** — взаимодействия признаков до заданной степени.
- **RBF**: $K(x,y) = \exp(-\gamma\lVert x-y\rVert^2)$ — similarity из евклидова расстояния, сильно нелинейные границы. Популярно в biomedical, но зависит от bandwidth $\gamma$ и переобучается в высокой размерности с шумом.

> [!warning] Single-kernel limitation
> Одно ядро не ловит одновременно глобальные тренды и локальную нелинейную структуру в гетерогенных данных. Для мультимодальных IoT healthcare данных ни одно ядро не адекватно → мотивация [[Multiple Kernel Learning]] и [[Adaptive Hybrid Kernel SVM]].
