---
title: Thesis Overview
type: note
tags: [thesis, overview, scope]
---

# Thesis Overview

> [!info] Background
> IoMT и wearable-сенсоры дают непрерывные высокоразмерные потоки данных (heart rate, SpO₂, ECG, accelerometer, ambient). Эти сигналы шумные и нестационарные — сенсоры различаются, среда вносит помехи, пациенты физиологически различны. Классические ML-методы, предполагающие чистые стационарные входы, плохо справляются.

Классификатор для clinical decision support должен:
1. справляться с высокой размерностью без переобучения;
2. быть устойчивым к шуму и выбросам;
3. адаптироваться к смене характеристик данных без полной перетренировки.

Особенно тяжёлый кейс — **elderly care**: несколько устройств одновременно, непостоянная adherence, меняющаяся со временем релевантность признаков.

## Почему SVM
[[SVM]] — разумная стартовая точка для высокоразмерной классификации (statistical learning theory, margin maximization, kernel trick). Проблема: стандартные SVM с **фиксированным** ядром плохо работают на шумных мультимодальных IoT-данных. Linear ловит глобальные тренды, но теряет локальные нелинейности; RBF моделирует локальную структуру, но переобучается на шуме в высокой размерности → см. [[Kernel Methods]].

## Scope
В рамках Master's timeline фокус на **одном** core-контрибьюшене: [[Adaptive Hybrid Kernel SVM]] = dynamic [[Feature Weighting]] + adaptive linear-RBF mixture.

Тезис будет:
- формулировать композитное ядро (global linear + local RBF, tunable $\mu$) на feature-weighted входах;
- разрабатывать механизм весов признаков из missingness + variance stability;
- давать полную математическую формулировку + псевдокод;
- оценивать на [[Datasets|OPPORTUNITY и CASAS]] против focused-набора [[Baselines]] по accuracy, F1, noise robustness.

> [!warning] Out of scope (→ [[Ethics and Future Work]])
> Online/streaming-адаптация весов на inference, edge-deployment, fairness-аудит по подгруппам, privacy-preserving (federated / differential privacy), deep-learning гибриды.
