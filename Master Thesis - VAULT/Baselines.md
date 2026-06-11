---
title: Baselines
type: note
tags: [thesis, baselines, experiments]
---

# Baselines

Три бейзлайна, изолирующие вклад hybrid kernel и [[Feature Weighting]].

| Baseline | Описание | Цель сравнения |
|---|---|---|
| **SVM-RBF** | Стандартный SVM с RBF-ядром; $(C,\gamma)$ через CV; без весов | Улучшает ли hybrid+weights самый частый single-kernel выбор |
| **SVM-Linear** | Стандартный SVM с linear-ядром; $C$ через CV; без весов | Добавляет ли RBF-компонент value поверх линейного бейзлайна |
| **Static MKL** | Выпуклая комбинация linear+RBF, $\mu$ через CV, **без** весов | Изолирует вклад feature weighting (ablation, [[Objectives and Hypotheses|H3]]) |

→ [[Experimental Protocol]]
