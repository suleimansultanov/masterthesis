---
title: Literature Review Notes
type: note
tags: [thesis, literature-review]
---

# Literature Review — структура

Три релевантных области (venues: IEEE, ACM, Elsevier, Springer, MDPI):

## 2.1 SVMs & kernel methods для HD biomedical
→ [[SVM]], [[Kernel Methods]]. SVM хорош в HD/small-sample; выбор ядра критичен; single-kernel не ловит global+local одновременно.

## 2.2 Multiple Kernel Learning
→ [[Multiple Kernel Learning]]. Полезен при гетерогенной структуре, но веса статичны на inference + игнор per-feature качества.

## 2.3 Noise handling & feature weighting в IoT healthcare
→ [[Noise Handling in IoT Healthcare]], [[Feature Weighting]]. Интеграция весов прямо в ядро — underexplored.

## 2.4 Summary of gaps
1. MKL учит статичные kernel-веса, не адаптируются на inference.
2. Feature weighting как decoupled препроцессинг → suboptimal взаимодействие с ядром.
3. Нет работы, совместно оптимизирующей adaptive mixture + data-quality веса в одном SVM для IoT healthcare.

→ полностью: [[Research Gap and Novelty]].
