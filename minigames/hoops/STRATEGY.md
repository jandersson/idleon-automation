# Hoops strategy notes

What the bot assumes about the game and how each piece of the make-shot
pipeline fits together. Living document — update as we learn more.

## Game-physics assumptions (empirical)

- **Launch is fixed-power and fixed-direction.** Same platform Y at click +
  same hoop position → same trajectory. Tuning is therefore deterministic
  per shot: for any given hoop, there's exactly one optimal `platform_y`.
- **Click position has no measurable effect** on aim (verified May 3 with
  the click_sweep and click_extreme experiments — both produced ball
  trajectories within the ~20px noise floor regardless of click_x or
  click_y). The bot clicks at the hoop position because that's a sane
  default; it could click anywhere.
- **Hoop respawns randomly between shots** at score 0–19 — both X and Y
  vary across the playfield.
- **At score ≥10**, the platform starts moving horizontally. We have
  `home_x` / `X_TOLERANCE` machinery for this but it's currently disabled
  (tolerance=9999) because we rarely get past score 10 today.
- **At score ≥20**, the hoop also moves between frames. Not handled.
- **At score ≥30**, the hoop moves both axes. Not handled.

## Scoring

- Direct rim-touched make = **1 pt**.
- Nothing-but-net (ball through rim without touching rim or backboard) = **2 pts**.
- Trophy at single-trial score **40+**.
- Pet from combined score **66** across 3 trials.

## How a shot works (end to end)

1. **Detect rim** — `find_rim` in `detector.py` template-matches `rim.png`
   against the right half of the frame at multiple scales. Returns
   (hoop_x, hoop_y) at conf ≥ 0.5.
2. **Pick offset** — `_compute_offset` calls a fitted `Predictor` that
   returns the predicted optimal `platform_y` for this hoop. Offset =
   target_y − hoop_y. Plus a perturbation if previous shots at the same
   hoop position missed (perturbation sweep, see below).
3. **Wait for platform** — sample `find_platform` until the platform
   crosses target_y going up (`REQUIRED_DIRECTION="up"`) within
   `Y_TOLERANCE=6` px.
4. **Fire** — `click(hoop_x, hoop_y)` (window-relative). 10–40 ms random
   delay before, ±3 px positional jitter.
5. **Capture flight** — frames written to
   `assets/monitor/shot_NNN_<ts>/flight_*.png` for 4 seconds.
6. **Detect make/miss** — three signals must agree:
   - **OCR** (`common.score_ocr.read_score`) — multi-pass tesseract
     voting on the score region, 4 passes (PSM 8, 10, inverted, dilated)
     require ≥ 2 agreement.
   - **Running score anchor** — make iff `score_after - session_score`
     is 1 or 2 (a real per-shot increment). The anchor advances even on
     ambiguous multi-shot OCR drops, so the live count stays accurate.
   - **Trajectory cross-check** —
     `abs(ball_x_at_rim_height - hoop_x) ≤ TRAJECTORY_MAKE_TOLERANCE (60 px)`.
     Rejects tesseract misreads where the ball clearly missed.
7. **Log** — one row to `assets/shots.db` with all the above plus
   diagnostic columns (predicted_offset, perturbation, code_commit,
   predictor_kind, ball_apex_y, etc.).
8. **Update perturbation tracking** — if missed, increment
   `misses_at_current_hoop` so the next shot at this same position
   tries a different offset perturbation.
9. **Snapshot + commit** — at session end, regenerate
   `assets/shots_snapshot.json` (gitignored DB → tracked aggregate)
   and auto-commit + push (within 09:00–22:00 Stockholm).

## Predictor

`PREDICTOR_KIND` in `main.py` selects between two implementations in
`common/predictor.py`:

- **`"knn"` (default)** — `KnnPredictor`. Inverse-distance-weighted KNN
  over past makes' (hoop_y, hoop_x, platform_y) tuples. K=5. Adapts to
  local curvature in the optimal-py surface; recommended.
- **`"bivariate"`** — `BivariatePredictor`. Closed-form OLS for
  `target_y = a·hoop_y + b·hoop_x + c`. Faster but biased in regions
  far from the training-data cluster — needed hardcoded overrides for
  low-hoop_x regions before we switched. Kept as A/B baseline.

`fit_knn(rows, k)` / `fit_bivariate(rows)` are the factory functions;
both take rows from `common.shot_log.fetch_makes(conn, direction)`.

## Perturbation sweep

Hoops only respawn after a make, so a stuck offset would burn every
life. After each consecutive miss at the same (hoop_x, hoop_y), the
next shot adds a perturbation to the predicted offset:

```
[0, -8, 8, -16, 16, -24, 24, -32, 32, -48, 48, -64, 64, -80, 80]
```

Make or hoop-position change resets the counter. Each perturbed shot
gets logged with the perturbation, so makes from explored offsets feed
back into the predictor on next session.

## Tuning knobs (rarely touched)

| Constant | Purpose | When to revisit |
|---|---|---|
| `PREDICTOR_KIND` | "knn" or "bivariate" | A/B comparing predictor variants |
| `Y_TOLERANCE` (6) | Px window around target_y to fire on `in_window` | If bot fires too often or never |
| `REQUIRED_DIRECTION` ("up") | Platform direction at fire | Have not had reason to flip since the dir=up regime was found |
| `PERTURBATION_SEQUENCE` | Miss-driven offset sweep | If stuck regions need wider exploration |
| `TRAJECTORY_MAKE_TOLERANCE` (60) | Px window for ball-at-rim-x to count as a make | If real makes are getting rejected as overshoots |
| `COLD_START_OFFSET` (20) | Used until ≥4 makes are in the DB for the predictor to fit | New install, or DB wiped |

## Open problems

1. **Score ≥10 moving platform** — not handled. The `home_x` machinery
   exists but is disabled. Will need re-enabling before we can clear
   the trophy (40+ score requires playing through 10/20/30 thresholds).
2. **Score ≥20 moving hoop** — bot uses a stale hoop position. Will
   need per-shot re-detection right before the platform crosses target.
3. **Tesseract is unreliable on tiny pixel-art digits.** Multi-pass
   voting cuts most misreads but not all. The durable fix is template
   matching against pre-extracted digit PNGs (see
   `docs/metrics_backlog.md` "Higher cost").

## Aim-quality backlog

Open ideas for raising the per-session make rate, ordered by expected
value-per-effort. Drop entries as they ship.

- **Hoop-confidence gating.** Skip the shot when the rim template-match
  confidence is low (e.g. `hoop_conf < 0.85`) — the rim was probably
  misdetected and shooting wastes a life. Concrete data: 2026-05-04
  session 23:52 burned 4 lives on a hoop detected at conf=0.57 before a
  perturbation finally landed; skipping it would have given those 4
  lives back to good shots later in the session. Cheapest win on the
  list.
- **GP-variance-scaled perturbation step.** GpPredictor exposes
  `predict_with_std`. The hoop only respawns on a make, so *skipping*
  uncertain shots is wrong — it just burns lives standing still. The
  useful move is to consume σ as a step-size signal: high σ (sparse
  region, predictor is guessing) → take bigger sweep steps (e.g.
  ±16, ±32, ±64) to walk into the make zone fast; low σ → keep the
  current ±8 fine-tuning. Same mechanism as the existing
  PERTURBATION_SEQUENCE, just adaptive.
- **Directional perturbation from ball trajectory.** Current sweep is
  symmetric (`-8, +8, -16, +16, ...`) and ignores where the ball
  actually landed. Each miss already logs `ball_x_at_rim_height` and
  `ball_landing_x`; if the ball was short, push offset one way; if
  long, the other. Higher information yield per miss than blind
  sweeping. Combines with the variance-scaled step above: σ sets the
  *magnitude* of the next perturbation, ball trajectory sets its
  *sign*.
- **Bigger sample base.** 58 clean makes (post-rattle filter,
  2026-05-05) train the GP. Every additional clean session shrinks the
  high-variance regions automatically; comes for free with play time.
  Not an action item, but worth remembering as the implicit baseline
  improvement against which other changes are measured.
