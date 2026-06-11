---
title: Feature Weighting
type: concept
tags: [concept, feature-weighting, noise]
---

# Feature Weighting

Каждое feature-измерение получает вес, отражающий информативность/надёжность. Традиционно веса берут из mutual information, Fisher ratio, variance thresholds (Zhang 2011; Yang 2005) — но **статично**, один раз на препроцессинге, без учёта temporal-изменений надёжности (частых в IoT). Веса, посчитанные независимо от ядра, могут плохо взаимодействовать с similarity-мерой (Lin & Wang 2004).

> [!tip] Вклад тезиса
> Веса вычисляются из data-quality индикаторов и применяются **внутри** kernel-функции → ненадёжные фичи меньше вносят в similarity. Формула:
> $$ w_k = (1 - m_k)\cdot \sigma(-\lambda s_k) $$
> где $m_k$ — missingness rate, $s_k = \sigma_k/(|\mu_k|+\varepsilon)$ — coefficient of variation. Полностью → [[Mathematical Formulation]].

Связано: [[Noise Handling in IoT Healthcare]], [[Adaptive Hybrid Kernel SVM]].
