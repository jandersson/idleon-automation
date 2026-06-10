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
  "the bot's eye": all four models fitted live in the browser on the
  real shot record. Hover to watch KNN pick its neighbours, scrub the
  GP lengthscale, see the σ map glow where the bot is guessing, and
  run the bob loop through the make-probability field to watch it
  pick its firing moment.

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
