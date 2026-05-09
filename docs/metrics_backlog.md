# shots.db metrics backlog (historical)

Open follow-ups have moved to GitHub issues — see
[`label:area:metrics`](https://github.com/jandersson/idleon-automation/issues?q=is%3Aissue+is%3Aopen+label%3A%22area%3Ametrics%22).
This file keeps the shipped diagnostic columns so the schema's
evolution stays browsable in one place.

## Shipped

### Cheap, high value

- [x] **`click_x`, `click_y`** — actual click coordinates, kept for
      diagnostic purposes even though click position no longer varies
      (see `cleanup_backlog.md`).
- [x] **`window_w`, `window_h`** — detect mid-session resize.
      Fraction-based regions silently shift if the user resizes the
      game window.
- [x] **`perturbation`** — the explicit `±N` px we applied. Currently
      derivable from `offset - predicted_offset` but only if you know
      the predictor at log time; logging directly makes "did the sweep
      find the make zone?" a one-line query.
- [x] **`lives_diff`** — already detected (the "lives counter ticked
      down" print), now persisted. Cross-check on "did this shot
      really miss" — score and life count occasionally disagree.

### Medium cost, high value

- [x] **Ball trajectory metrics** — `ball_x_at_rim_height`,
      `ball_apex_y`, `ball_landing_x` from the per-shot flight frames.
      Lets us classify misses (undershoot/overshoot/vertically-off)
      automatically without grepping monitor folders. HSV mask + a
      tiny tracking script over the existing flight frame sequence.

### Diagnostic columns

- [x] **`predicted_offset`** — what the predictor would have returned,
      separate from the override-applied `offset`. Lets us track when
      the predictor catches up enough to retire each override.
- [x] **`code_commit`** — short git hash of the working tree at session
      start, with `-dirty` suffix if uncommitted. Correlates make-rate
      changes with code changes.
- [x] **`predictor_kind`** — which predictor produced this shot's
      target_y (`"knn"`, `"bivariate"`, or NULL for cold-start). Lets
      us A/B compare predictors on the same dataset.

### Higher cost

- [x] **Absolute score (OCR'd)** — instead of `made` boolean, read the
      actual score number. Tells us 1-pt vs 2-pt ("nothing but net" per
      wiki). Could nudge calibration toward swish-quality shots later.
      Implemented via pytesseract; requires the tesseract binary on
      PATH (`winget install --id=UB-Mannheim.TesseractOCR`). Falls back
      to NULL when missing. Multi-pass voting (2026-05-03) cuts misreads.
