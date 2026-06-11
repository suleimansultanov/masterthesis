---
title: Experimental Protocol
type: note
tags: [thesis, experiments, evaluation]
---

# Experimental Protocol

## Cross-validation
5-fold **stratified** CV для подбора гиперпараметров на train (стратификация сохраняет распределение классов). Для устойчивости весь эксперимент (split + CV + test) повторяется **5 раз** с разными seeds → результаты как **mean ± std**.

## Primary metrics
- **Accuracy** — общая корректность.
- **Macro-averaged F1** — учитывает дисбаланс (усреднение F1 по классам с равным весом).

## Noise robustness test
Гауссовский шум $\sigma \in \{0.1, 0.3, 0.5, 0.7, 1.0\}$ инжектируется в случайные **30%** feature-измерений на test-time. Строятся performance degradation curves (accuracy и F1 vs noise level) для каждого метода. Симулирует деградацию/отказ сенсоров в деплое.

## Ablation study
Полная модель (hybrid kernel + feature weights) против **Static MKL** (hybrid kernel без весов) → изолирует вклад data-quality-driven весов. Прямой тест [[Objectives and Hypotheses|H3]].

## Statistical significance
Paired **Wilcoxon signed-rank** по $5 \times 5 = 25$ fold-результатам; различия с каждым бейзлайном на уровне $p < 0.05$.

→ Связано: [[Baselines]], [[Datasets]].
