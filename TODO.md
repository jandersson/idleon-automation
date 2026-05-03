# TODO

The detailed work-in-progress lists live in `docs/`:

- **`docs/cleanup_backlog.md`** — refactors and dead-code removal
- **`docs/metrics_backlog.md`** — `shots.db` columns to add, OCR work

This file holds only items that don't fit those buckets.

## Open

- **Hoops scoring 10+** — when the in-game score reaches 10 the platform
  starts moving horizontally. The `home_x` / `X_TOLERANCE` machinery is
  scaffolded but disabled (tolerance=9999). Re-enable + tune before
  attempting the 40+ trophy run.
- **Hoops scoring 20+** — hoop also moves between frames. Currently we
  read hoop position once per shot; for moving hoops we'd need to
  re-detect right before the platform reaches target_y.
- **Catching minigame** — scaffold only; detectors return None. Need
  fly + hoop-gap detectors before this runs.
- **Chopping** — last verified state was "button click is suspect".
  Hasn't been touched since the early-May Hoops focus push. Needs a
  fresh look + a careful single-attempt re-test.
- **Darts** — release-pose template matching works, score-region diff
  per throw works. Multi-template per spawn-height (the dominant
  accuracy variable) not yet built.
- **Predictor: LOESS (locally-weighted linear regression)** — sits
  between current KNN (local average, no slope) and Bivariate (global
  slope, no locality). For each query, fit a small weighted lstsq on
  the K nearest past makes with weights = 1/distance. Add as
  `LoessPredictor` in `common/predictor.py` + `PREDICTOR_KIND="loess"`
  switch in `minigames/hoops/main.py`. Uses the same `fetch_makes`
  data — no new schema. Compare make-rate vs current `knn`.

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
