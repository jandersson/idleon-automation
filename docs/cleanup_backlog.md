# Cleanup backlog

Code that was added during investigation/experimentation and is now safe to
delete. Items here exist to keep them tracked and not forgotten.

## Hoops: click-position machinery

**Conclusion from experiments (2025-05-03):** click position has no
measurable effect on ball trajectory.

- click_y sweep (170 → 490, 320px range) → ball_x_at_rim variation: 24px
  (within the ~20px noise floor at fixed click).
- click_extreme (window corners + center vs at-hoop) — ran the experiment
  and didn't see meaningful trajectory differences either.

The earlier "click matters" reading was confounded by perturbation-sweep
luck and a small number of flukey makes. The clean isolation experiments
disproved it.

### Remove (done 2026-05-03)

- [x] `CLICK_STRATEGY` constants and all branches in `_pick_click_position`
- [x] `VARIED_CLICK_POSITIONS`, `CLICK_SWEEP_Y_OFFSETS`,
      `CLICK_EXTREME_POSITIONS` constants
- [x] `EXPERIMENT_MODE` env-var dispatch in `minigames/hoops/main.py`;
      removed the two click experiments from the hoops launcher entry
      (kept the general `experiments` slot for future use)
- [x] `tests/test_click_strategy.py` deleted

### Follow-ups (still open)

- [ ] **DB rows from the click-sweep / click-extreme runs** — still
      labelled `made=0` correctly thanks to OCR+trajectory validation.
      Adding a `experiment_label` column would let the predictor exclude
      them in case future false-positives slip through. Low priority.
- [x] **CLAUDE.md note** — added a "Hoops: established findings" section
      under Architecture covering click position, OCR + trajectory cross-
      check, and pre-shot OCR flakiness.
- [ ] **Tune `TRAJECTORY_MAKE_TOLERANCE` (currently 60px)** — only
      relevant if we see a real make get rejected because the ball
      detector mistracked. Worth revisiting after a few sessions of
      data with the new validation.

### Done 2026-05-03

- [x] `_pick_click_position` inlined at the call site (was a one-liner)
- [x] Launcher's `experiments` config slot removed (was empty, plus
      its `extra_env` plumbing in `_start_bot` / `_spawn`)
- [x] Stopped logging `click_x` / `click_y` to `shots.db`. Removed from
      schema, migration, dump_shots query, and tests. Old DBs still
      have the columns sitting there harmlessly (no DROP COLUMN
      migration — they take ~16 bytes per row, not worth the churn).

### Keep

- `click_x` / `click_y` columns in `shots.db` — still useful per-shot
  diagnostic info even if we don't vary them.
- Launcher's general `experiments` config slot — useful framework for
  future tests (just not these specific ones).

## Refactor: split predictors into their own module (done 2026-05-03)

- [x] New `common/predictor.py` holds `Predictor` protocol +
      `KnnPredictor` + `BivariatePredictor` + `fit_knn` + `fit_bivariate`
- [x] `common/shot_log.py` shrunk: removed `KnnPredictor` and
      `fit_target_predictor`. Added `fetch_makes(conn, dir)` which
      returns the rows; callers pick a predictor and fit themselves.
- [x] `minigames/hoops/main.py` has a `PREDICTOR_KIND` constant
      (`"knn"` or `"bivariate"`); call site picks the right `fit_*`.
- [x] `predictor_kind` column added to `shots.db` so per-shot history
      records which predictor produced the target_y.
- [x] Tests split: `tests/test_predictor.py` for the new module,
      `tests/test_shot_log.py` slimmed to just logging concerns.
