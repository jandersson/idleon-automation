---
title: idleon-automation docs
---

# idleon-automation — docs

Companion notes and tools for the screen-reading bots at
[github.com/jandersson/idleon-automation](https://github.com/jandersson/idleon-automation).

## Predictors

- **[predictors.md](predictors.md)** — refresher notes on the
  prediction algorithms in `common/predictor.py`: KNN with
  inverse-distance weighting, bivariate OLS, GP regression (the
  default), the GP *classifier* behind hoops' `make_prob`
  candidate-ranking, and the planned darts E[stripe] model with its
  2D wind encoding.
- **[predictor_playground.html](predictor_playground.html)** —
  interactive heatmap visualisation. Switch between predictors,
  scrub K and noise, hover for predicted vs true offset.

## Investigations

- **[hoops_findings.md](hoops_findings.md)** — the June 2026 miss
  investigation: why misses were never aim error (structure clanks +
  launch-velocity physics), the per-hoop direction policy, multi-signal
  make detection, and the measured outcome (25.5% → 34.8% baseline,
  then the make_prob classifier beyond it).

## Backlogs

- **[cleanup_backlog.md](cleanup_backlog.md)** — refactors and
  dead-code removal.
- **[metrics_backlog.md](metrics_backlog.md)** — `shots.db` columns
  to add, OCR work, instrumentation ideas.
