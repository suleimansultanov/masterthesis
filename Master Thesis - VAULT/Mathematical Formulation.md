---
title: Mathematical Formulation
type: note
tags: [thesis, math, kernel, optimization]
---

# Mathematical Formulation

## Composite kernel
Пусть $x \in \mathbb{R}^d$ — входной сэмпл. Композитное ядро:

$$
K(x_i, x_j) = \mu \cdot K_{lin}(W \odot x_i,\, W \odot x_j) + (1-\mu)\cdot K_{rbf}(W \odot x_i,\, W \odot x_j)
$$

- **Feature weight vector** $W = (w_1, \dots, w_d)$, $w_k \in [0,1]$ — оценка надёжности признака $k$. $W \odot x$ — поэлементное (Hadamard) умножение, масштабирует фичу её весом **до** ядра.
- **Mixture coefficient** $\mu \in [0,1]$: $\mu=1$ → feature-weighted linear SVM; $\mu=0$ → feature-weighted RBF SVM; промежуточные значения смешивают.
- **Linear kernel**: $K_{lin}(a,b) = a^\top b$ — глобальные линейные связи.
- **RBF kernel**: $K_{rbf}(a,b) = \exp(-\gamma \lVert a-b \rVert^2)$ — локальная similarity, $\gamma$ задаёт масштаб локальности.

> [!important] Почему "внутри" ядра
> Веса применяются **до** вычисления ядра → ненадёжная фича модулирует и distance, и inner product. Фича с $w \approx 0$ практически невидима обоим компонентам; $w=1$ вносит вклад в естественном масштабе. Это **не** препроцессинговая feature selection.

## Feature weight computation → [[Feature Weighting]]
**Missingness rate** для фичи $k$:
$$ m_k = \frac{\#\{\text{missing/invalid в фиче } k\}}{\#\{\text{train samples}\}} $$

**Variance stability** (coefficient of variation):
$$ s_k = \frac{\sigma_k}{|\mu_k| + \varepsilon} $$
($\varepsilon$ — малая константа против деления на ноль при near-zero mean).

**Combined weight**:
$$ w_k = (1 - m_k)\cdot \sigma\!\left(-\lambda \cdot s_k\right), \qquad \sigma(z) = \frac{1}{1+e^{-z}} $$
$\lambda > 0$ — sensitivity hyperparameter (насколько агрессивно штрафуется variance-нестабильность).

> [!note] Мультипликативная структура
> $(1-m_k)\cdot\sigma(-\lambda s_k)$ держит индикаторы независимыми: фича с 80% пропусков получает максимум $0.2$ независимо от стабильности дисперсии. После — нормализация $\sum_k w_k = d$ (сохраняет масштаб → $C$ и $\gamma$ интерпретируемы как в обычном SVM).

## SVM optimization (dual)
Train set $\{(x_1,y_1),\dots,(x_n,y_n)\}$, $y_i \in \{-1,+1\}$:
$$ \max_\alpha\; L(\alpha) = \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j}\alpha_i \alpha_j y_i y_j K(x_i,x_j) $$
$$ \text{s.t. } 0 \le \alpha_i \le C \;\forall i, \quad \sum_i \alpha_i y_i = 0 $$

Decision function:
$$ f(x) = \operatorname{sign}\!\left(\sum_i \alpha_i y_i K(x_i, x) + b\right) $$

> [!tip] Design choice
> $W$ вычисляется **до** построения kernel matrix и **не** оптимизируется совместно с $\alpha$. Это держит $K$ положительно полуопределённой (неотрицательная взвешенная сумма валидных ядер) → стандартная выпуклая QP (LIBSVM через `SVC(kernel='precomputed')`). Избегаем non-convex совместной оптимизации.

## Hyperparameter search (grid + CV) → [[Experimental Protocol]]
- $\mu \in \{0.0, 0.1, \dots, 1.0\}$ — 11 значений
- $C \in \{0.01, 0.1, 1, 10, 100\}$ — 5
- $\gamma \in \{0.001, 0.01, 0.1, 1, 10\}$ — 5
- $\lambda \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$ — 5

Итого $11 \times 5 \times 5 \times 5 = 1{,}375$ комбинаций, 5-fold CV. Выбирается max mean CV-accuracy на train.

## Pseudocode
**Algorithm 1 — Training** (вход: $X \in \mathbb{R}^{n\times d}$, $y$; гиперпараметры $\mu, C, \gamma, \lambda$):
1. **Data-quality indicators**: для каждой фичи `m_k = count_missing(X[:,k])/n`; `s_k = std(X[:,k]) / (|mean(X[:,k])| + ε)`
2. **Feature weights**: `w_k = (1 − m_k)·sigmoid(−λ·s_k)`; нормализация `w ← w·(d / Σ w_k)`
3. **Feature scaling**: `X̃ ← X ⊙ W`
4. **Composite kernel matrix**:
   - `K_lin ← X̃ · X̃ᵀ`
   - `K_rbf[i,j] ← exp(−γ·‖X̃[i] − X̃[j]‖²)`
   - `K ← μ·K_lin + (1 − μ)·K_rbf`
5. **Solve dual**: `(α*, b*) ← SVM_Dual_Solver(K, y, C)`
6. **Return** `(support_vectors, α*, b*, W, μ, γ)`

**Algorithm 2 — Prediction** (вход: `x_new`, модель):
1. `x̃_new ← W ⊙ x_new`
2. для каждого SV: `k_i ← μ·(x̃_iᵀ x̃_new) + (1−μ)·exp(−γ‖x̃_i − x̃_new‖²)`
3. `y_pred ← sign(Σ_i α*_i · y_i · k_i + b*)`
