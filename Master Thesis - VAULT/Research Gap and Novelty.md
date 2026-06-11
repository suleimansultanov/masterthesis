---
title: Research Gap and Novelty
type: note
tags: [thesis, gap, novelty]
---

# Research Gap and Novelty

Две линии исследований развивались **независимо**:
- [[Multiple Kernel Learning|MKL]] — выпуклая комбинация базовых ядер;
- weighted SVM — разные веса разным feature-измерениям для борьбы с шумом/дисбалансом.

Их интеграция именно в IoT healthcare почти не изучена.

> [!success] Novelty statement
> Ни один известный подход не решает совместно: **(1)** data-quality-driven per-feature адаптацию весов, встроенную прямо в вычисление ядра (веса реагируют на измеримые индикаторы надёжности сенсора — missingness, variance stability); **(2)** adaptive fusion глобального (linear) и локального (RBF) компонентов; **(3)** прицельное применение к шумным высокоразмерным IoT healthcare потокам.

## Чем отличается от существующего
| Подход | Ограничение |
|---|---|
| Standard MKL (Gönen & Alpaydin 2011; Rakotomamonjy 2008) | веса комбинации ядер фиксируются на train, не обновляются на inference |
| Feature-weighted SVM (Zhang 2011; Yang 2005) | статические веса из препроцессинга (MI, variance thresholds), не связаны с композицией ядра |
| Fuzzy SVM (Lin & Wang 2004) | membership-степени сэмплам, но per-feature data-quality не идёт в ядро |

> [!tip] Ключевая идея
> Веса признаков **модулируют вычисление ядра напрямую** — ненадёжные признаки де-эмфазируются *внутри* similarity-меры, а не отфильтровываются заранее.

См. сводку: [[Multiple Kernel Learning]], [[Feature Weighting]].
