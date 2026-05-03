# shots.db metrics backlog

Suggestions for additional columns / data to log per hoops shot. Ordered by
cost vs payoff. Drop entries as they ship.

## Cheap, high value

- [x] **`click_x`, `click_y`** — actual click coordinates. We just confirmed
      click position affects aim; logging it lets us validate AND experiment
      with offsets from the hoop (above/below/relative).
- [x] **`window_w`, `window_h`** — detect mid-session resize. Fraction-based
      regions silently shift if the user resizes the game window.
- [x] **`perturbation`** — the explicit `±N` px we applied. Currently
      derivable from `offset - predicted_offset` but only if you know the
      predictor at log time. Logging directly makes "did the sweep find the
      make zone?" a one-line query.
- [x] **`lives_diff`** — already detected (the "lives counter ticked down"
      print), just not persisted. Cross-check on "did this shot really
      miss" — score and life count occasionally disagree.

## Medium cost, high value

- [x] **Ball trajectory metrics** — extract `ball_x_at_rim_height`,
      `ball_apex_y`, `ball_landing_x` from the per-shot flight frames.
      Lets us classify misses (undershoot/overshoot/vertically-off)
      automatically without grepping monitor folders. HSV mask + a tiny
      tracking script over the existing flight frame sequence.

## Diagnostic columns (open)

- [x] **`predicted_offset`** — what the predictor would have returned,
      separate from the override-applied `offset`. Lets us track when
      the predictor catches up enough to retire each override.
- [x] **`code_commit`** — short git hash of the working tree at session
      start, with `-dirty` suffix if uncommitted. Correlates make-rate
      changes with code changes.
- [ ] **`override_applied`** (TEXT) — which override (if any) determined
      the offset for this shot. Values like `low_x_high_hoop`,
      `low_x_low_hoop`, `mid_x_low_hoop`, or NULL. Easier query than
      reconstructing the override conditions every time.
- [ ] **`rim_match_scale`** (REAL) — the scale that
      `match_multiscale_center` picked for the rim template. Currently
      the `_scale` return value is discarded. Catches silent window-
      resize regressions (a sudden shift in scale across shots in one
      session means the window changed size).
- [ ] **`time_since_last_shot_ms`** (INTEGER) — derivable from `fired_at`
      but explicit avoids parsing in queries. Useful for "was this shot
      in the post-make-respawn window or after a stuck-hoop sequence?"
- [ ] **`bob_period_ms`** (INTEGER) — time between consecutive `py`
      peaks in the platform sample buffer. Once score ≥ 10 the
      platform starts moving and the bob shape changes; this would let
      us see when.
- [ ] **`ball_flight_ms`** (INTEGER) — first-detected to last-detected
      time across flight frames. A real make has a longer flight than
      an undershoot (ball passes through net vs disappears off-screen).
      Could be a third make signal alongside OCR + trajectory x.

## Higher cost

- [x] **Absolute score (OCR'd)** — instead of `made` boolean, read the
      actual score number. Tells us 1-pt vs 2-pt ("nothing but net" per
      wiki). Could nudge calibration toward swish-quality shots later.
      Implemented via pytesseract; requires the tesseract binary on PATH
      (`winget install --id=UB-Mannheim.TesseractOCR`). Falls back to
      NULL when missing. Multi-pass voting (2025-05-03) cuts misreads.

- [ ] **Replace tesseract with template-matched digits 0-9** — Idleon's
      score font is fixed-pixel pixel-art, not what tesseract was trained
      on. Pre-extract one PNG per digit and `cv2.matchTemplate` for
      effectively-100% reliable score reads. Same pattern we use for the
      rim. Currently we have data with scores up to 7, so 0-7 templates
      could be auto-extracted; 8-9 would need samples. Drop pytesseract
      dependency once done.
- [ ] **Bob range snapshot** — `bob_ymin`, `bob_ymax` at fire time. Right
      now we assume ~290-510, but per the wiki the platform starts moving
      at score 10+ and the bob may shift in late game.
