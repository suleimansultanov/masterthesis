---
title: Timeline and Milestones
type: note
tags: [thesis, timeline, project-management]
---

# Timeline and Milestones

16-недельный (4 мес.) Master's проект. Буфер на debugging; writing стартует рано.

| Phase | Weeks | Activities / deliverables |
|---|---|---|
| **1. Data prep & lit finalization** | 1–3 | Acquire + preprocess OPPORTUNITY/CASAS; feature extraction pipelines; синтетические noise-варианты; финализация lit review. **Deliverable**: cleaned datasets, preprocessing code, noise injection module. |
| **2. Core model implementation** | 4–7 | Модуль весов (missingness + variance stability); adaptive hybrid kernel с precomputed matrix; интеграция со scikit-learn; grid search pipeline; все три [[Baselines]]. **Deliverable**: working model, baselines, initial results. |
| **3. Experiments & evaluation** | 8–11 | Full CV (5 seeds × 5 folds) на обоих датасетах; noise robustness; ablation; significance testing; таблицы/фигуры. **Deliverable**: полные результаты, анализ. |
| **4. Thesis writing & defense** | 12–16 | Methodology/results/discussion; визуализации; правки по фидбеку супервизора; финальный манускрипт + defense slides. |

> [!note] Meetings
> Biweekly progress meetings с супервизором ([[Admin - Thesis Registration|Prof. Iliev]]); краткий progress report к каждой. Writing идёт параллельно с имплементацией.
