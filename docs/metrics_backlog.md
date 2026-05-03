# shots.db metrics backlog

Suggestions for additional columns / data to log per hoops shot. Ordered by
cost vs payoff. Drop entries as they ship.

## Cheap, high value

- [x] **`click_x`, `click_y`** — actual click coordinates. We just confirmed
      click position affects aim; logging it lets us validate AND experiment
      with offsets from the hoop (above/below/relative).
- [ ] **`window_w`, `window_h`** — detect mid-session resize. Fraction-based
      regions silently shift if the user resizes the game window.
- [x] **`perturbation`** — the explicit `±N` px we applied. Currently
      derivable from `offset - predicted_offset` but only if you know the
      predictor at log time. Logging directly makes "did the sweep find the
      make zone?" a one-line query.
- [x] **`lives_diff`** — already detected (the "lives counter ticked down"
      print), just not persisted. Cross-check on "did this shot really
      miss" — score and life count occasionally disagree.

## Medium cost, high value

- [ ] **Ball trajectory metrics** — extract `ball_x_at_rim_height`,
      `ball_apex_y`, `ball_landing_x` from the per-shot flight frames.
      Lets us classify misses (undershoot/overshoot/vertically-off)
      automatically without grepping monitor folders. HSV mask + a tiny
      tracking script over the existing flight frame sequence.

## Higher cost

- [ ] **Absolute score (OCR'd)** — instead of `made` boolean, read the
      actual score number. Tells us 1-pt vs 2-pt ("nothing but net" per
      wiki). Could nudge calibration toward swish-quality shots later.
- [ ] **Bob range snapshot** — `bob_ymin`, `bob_ymax` at fire time. Right
      now we assume ~290-510, but per the wiki the platform starts moving
      at score 10+ and the bob may shift in late game.
