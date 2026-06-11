---
title: SVM
type: concept
tags: [concept, svm, machine-learning]
---

# Support Vector Machine (SVM)

Метод классификации/регрессии для высокоразмерных пространств. Идея — построить **maximum-margin hyperplane**, разделяющую классы с наибольшим зазором → generalization-гарантии через теорию Vapnik-Chervonenkis. [[Kernel Methods|Kernel trick]] расширяет до нелинейной классификации.

Хорош в режиме high-dimensional, small-sample (типично для клиники). Конкурентен deep learning при ограниченных данных или когда важна интерпретируемость (Vallabhuni & Debasis 2025; Wang et al. 2023).

**Dual objective** и decision function → см. [[Mathematical Formulation]].

Связано: [[Kernel Methods]], [[Multiple Kernel Learning]], [[Adaptive Hybrid Kernel SVM]].
