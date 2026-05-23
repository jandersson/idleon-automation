# TODO

Most open work is tracked as GitHub issues:
<https://github.com/jandersson/idleon-automation/issues>. Useful
labels:

- [`area:cleanup`](https://github.com/jandersson/idleon-automation/issues?q=is%3Aissue+is%3Aopen+label%3A%22area%3Acleanup%22) — refactors, dead-code removal, code follow-ups
- [`area:metrics`](https://github.com/jandersson/idleon-automation/issues?q=is%3Aissue+is%3Aopen+label%3A%22area%3Ametrics%22) — `shots.db` schema, OCR replacement
- [`minigame:hoops`](https://github.com/jandersson/idleon-automation/issues?q=is%3Aissue+is%3Aopen+label%3A%22minigame%3Ahoops%22) / [`minigame:mining`](https://github.com/jandersson/idleon-automation/issues?q=is%3Aissue+is%3Aopen+label%3A%22minigame%3Amining%22) — per-minigame work

`docs/cleanup_backlog.md` and `docs/metrics_backlog.md` keep the
historical record of shipped diagnostic columns and settled
experiments.

This file holds cross-cutting items, scheduled-agent IDs, and the
recent-changes snapshot that don't fit a single issue.

## Open

- **Hoops scoring 10+** — at score ≥10 the HOOP starts drifting
  horizontally between shots (wiki had this wrong; corrected from live
  observation 2026-05-23). Bot reads hoop position once per shot when
  `target_y` is None, so by fire time it's stale. Fix: re-detect hoop
  right before the platform crosses target_y, recompute target_y off
  the fresh position. Same fix likely covers score-20 and score-30
  escalations since they're just more-aggressive versions of the
  same motion.
- **Catching minigame** — scaffold only; detectors return None. Need
  fly + hoop-gap detectors before this runs.
- **Chopping** — last verified state was "button click is suspect".
  Hasn't been touched since the early-May Hoops focus push. Needs a
  fresh look + a careful single-attempt re-test.
- **Darts** — release-pose template matching works, score-region diff
  per throw works. Multi-template per spawn-height (the dominant
  accuracy variable) not yet built.
- **Predictor: Gaussian Process regression** — biggest jump available
  with the data we already have. Beyond just predicting platform_y,
  GP gives a *posterior variance* per query — the bot can say "high
  confidence, fire" vs "no nearby training data, skip this shot".
  That turns the predictor from a point estimator into a strategic
  signal. Reference implementation in own repo
  https://github.com/jandersson/gp_regression. Squared-exponential
  (RBF) kernel is the standard starting choice; lengthscale +
  variance hyperparameters fitted via marginal-likelihood
  maximisation. Add as `GpPredictor` in `common/predictor.py` +
  `PREDICTOR_KIND="gp"` switch. Likely needs sklearn or GPy; check
  pyproject.toml dependency policy first. New shot_log column for
  `predicted_uncertainty` so we can analyse the relationship between
  variance and actual hit/miss rate.

- **Predictor: LOESS / locally-weighted linear regression** — smaller
  upgrade on the current "KNN with inverse-distance" predictor,
  which is basically Nadaraya-Watson kernel regression with a hard
  K-nearest cutoff. Same weighted neighborhood, but instead of
  *averaging* the past makes' platform_y values, fit a small
  weighted lstsq through them so the predictor captures the local
  *slope* of the surface, not just the local height. Add as
  `LoessPredictor` in `common/predictor.py` + `PREDICTOR_KIND="loess"`
  switch in `minigames/hoops/main.py`. Uses the same `fetch_makes`
  data — no new schema. Cheaper to land than GP.

## Someday

- **Tighten `common.window.get_bounds`** so it doesn't match the first
  window containing the substring. Today it works because every
  minigame uses the unique "Legends Of Idleon" title, but if a future
  window title contains that string the matcher silently picks the
  wrong one. Could prefer exact match, or filter by process name.

## Scheduled agents

- **Weekly review** — `trig_01M2YTDKxnUyB9bYJpSZhiEc`. Fires every
  Saturday 10:00 Stockholm. Reads `shots_snapshot.json`, opens a PR
  with stats + suggestions if anything looks off. First run: 2026-05-09.
- **One-time deep review** — `trig_016jn83AvPZwZQJZjZpHHKgk`. Fires
  once on **2026-05-16 10:00 Stockholm**. Deeper analysis with
  predictor residuals.

Manage at: https://claude.ai/code/routines

## Done (snapshot, May 2026)

The bulk of recent work happened in early May 2026. Major arcs:

- **Hoops detection rewrite** — rim-only template (was full-structure
  with stretchy pole), tighter score region, 20s exit timeout when no
  rim found.
- **Make detection** — replaced fragile `score_changed` diff with
  multi-pass Tesseract OCR + ball-trajectory cross-check. Anchored on
  a running `session_score` so per-shot OCR drops don't lose makes.
- **Predictor split** — `common/predictor.py` holds `KnnPredictor` and
  `BivariatePredictor` (selectable via `PREDICTOR_KIND` in
  `minigames/hoops/main.py`). Replaces the older bivariate-only model
  with hardcoded overrides.
- **Click-position experiments** — proved click position has no
  measurable effect on aim. Removed all related machinery.
- **Auto pipeline** — every session ends with a snapshot refresh +
  auto-commit + auto-push (within 09:00–22:00 Stockholm). Predictor
  retrains from the latest committed data on every launch.
- **Launcher** — Bots tab + Setup tab + Frames tab; tries-counter
  strip auto-reads from the local Idleon save (via plyvel-ci +
  vendored Haxe decoder).
