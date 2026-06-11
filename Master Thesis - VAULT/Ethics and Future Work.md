---
title: Ethics and Future Work
type: note
tags: [thesis, ethics, future-work]
---

# Ethics and Future Work

## Ethical considerations
- **Data ethics & privacy** — только публичные анонимизированные бенчмарки (OPPORTUNITY, CASAS), без PII, доп. этическое одобрение не требуется. Privacy-by-design: локальное хранение, без cloud-загрузки raw data, документация provenance.
- **Transparency & reproducibility** — все процедуры, гиперпараметры, seeds, препроцессинг документированы и сдаются с тезисом.
- **Limitations & responsible reporting** — лимиты репортятся явно (где не бьёт бейзлайны, чувствительность к гиперпараметрам, допущения о шуме); uncertainty (std) + significance testing. **Никаких clinical effectiveness claims** — оценка офлайн на бенчмарках.
- **Broader impact** — реальный деплой потребовал бы patient consent, fairness по подгруппам, regulatory approval; вне scope.

## Future Work
> [!abstract] Расширения за пределами тезиса
> - **Online/streaming adaptation** — обновление весов и $\mu$ онлайн с приходом данных (incremental updates, сохраняя PSD kernel matrix).
> - **Edge deployment** — оптимизация под latency/memory: random Fourier features для RBF, SV pruning, квантование kernel matrix.
> - **Fairness & bias analysis** — консистентность по подгруппам (age, gender, device type), fairness-aware веса.
> - **Joint optimization** — совместная оптимизация весов, $\mu$ и SVM-параметров (ценой non-convexity).
> - **Additional kernel components** — polynomial / sigmoid поверх linear+RBF.
