---
title: idleon-automation docs
---

# idleon-automation — docs

Companion notes and tools for the screen-reading bots at
[github.com/jandersson/idleon-automation](https://github.com/jandersson/idleon-automation).

## Predictors

- **[predictors.md](predictors.md)** — refresher notes on the
  prediction algorithms used (or planned) in `common/predictor.py`:
  KNN with inverse-distance weighting, bivariate OLS, and the
  in-progress Gaussian Process upgrade.
- **[predictor_playground.html](predictor_playground.html)** —
  interactive heatmap visualisation. Switch between predictors,
  scrub K and noise, hover for predicted vs true offset.

## Backlogs

- **[cleanup_backlog.md](cleanup_backlog.md)** — refactors and
  dead-code removal.
- **[metrics_backlog.md](metrics_backlog.md)** — `shots.db` columns
  to add, OCR work, instrumentation ideas.
