---
title: Noise Handling in IoT Healthcare
type: concept
tags: [concept, noise, iot-healthcare]
---

# Noise Handling in IoT Healthcare

Шум, sensor drift, missingness — неотъемлемы для IoT healthcare данных:
- Al-amri et al. (2021) — noise handling как центральный челлендж ML в IoT (calibration errors, wireless artifacts, environmental interference).
- AL-Dhief et al. (2020) — variable signal quality в voice pathology surveillance.
- Talaat & El-Balka (2023) — motion artifacts + electrode displacement в wearable stress monitoring.

## Стратегии
- **Preprocessing-уровень**: robust normalization, outlier detection, missing value imputation (Abdellatif 2022 — учитывать noise-характеристики по модальности).
- **Inherently robust learning** (Mahdavinejad 2018) — не полагаться только на препроцессинг.
- **[[Feature Weighting]]** — более принципиальный подход: вес по надёжности/информативности.

> [!note]
> Интеграция feature weighting прямо в kernel evaluation (надёжность модулирует similarity) — underexplored; именно это направление для IoT healthcare берёт [[Adaptive Hybrid Kernel SVM]].
