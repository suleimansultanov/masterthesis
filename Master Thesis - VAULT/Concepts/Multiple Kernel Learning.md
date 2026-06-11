---
title: Multiple Kernel Learning
type: concept
tags: [concept, mkl, kernel]
aliases: [MKL]
---

# Multiple Kernel Learning (MKL)

Решает ограничение single-kernel, обучая комбинацию базовых ядер:
$$ K = \sum_m \beta_m K_m, \qquad \beta_m \ge 0,\ \sum_m \beta_m = 1 $$
Веса $\beta_m$ учатся совместно с SVM-параметрами.

- **Gönen & Alpaydin (2011)** — обзор MKL (по optimization approach, форме комбинации, методологии); MKL обходит single-kernel при гетерогенной структуре.
- **Rakotomamonjy et al. (2008) — SimpleMKL** — gradient descent по весам ядер с L1-constraint, конкурентен и дешевле в обучении.
- Полезно в biomedical/IoT (Wang 2023; Al-amri 2021; Muriira 2018).

> [!warning] Ограничения для streaming IoT
> 1. Веса комбинации фиксируются на train, **не адаптируются** на inference (Gönen & Alpaydin 2011; Sonnenburg 2006).
> 2. Работает на уровне ядра, игнорирует **per-feature** различия в качестве данных — шумная фича вносит равный вклад во все ядра.
>
> → мотивация интегрировать [[Feature Weighting]] в MKL → [[Adaptive Hybrid Kernel SVM]].
