# Hoops strategy notes

What the bot assumes about the game and how each piece of the make-shot
pipeline fits together. Living document — update as we learn more.

## Cost and cadence (player budget)

Researched 2026-05-05 from idleon.wiki, idle-skilling.fandom.com, and
community guides. Numbers are rounded best-effort — public sources are
thin and the wiki returned 403 to scraping; spot-check in-game when
you care about specifics.

- **Daily scored attempts: 3.** After the third the game still lets you
  play freely in **practice mode** with no rewards. Practice resets at
  daily reset.
  [wiki](https://idleon.wiki/wiki/Hoops_Minigame),
  [oreateai](https://www.oreateai.com/blog/mastering-the-hoops-minigame-in-idleon-tips-and-tricks/01505c7236c123c6eafec083b8fe60b0)
- **No real-time cooldown** between games. Game over → results screen →
  immediate restart. Bot can spin sessions back-to-back.
- **No play cost.** Free attempts after the one-time unlock puzzle.
  No tickets, no energy.
- **Practical implication for this bot**: after the 3 daily scored
  tries are burned, **practice mode is an unlimited budget** for
  growing `shots.db` — no resource cost to building the prior. Run
  freely.
- **In-game difficulty escalation** within a single attempt (corrected
  2026-05-23 from live observation; the wiki's framing had the moving
  subject wrong):
  - Score ≥10 → hoop starts moving horizontally (subtle, between shots)
  - Score ≥20 → hoop pans more aggressively / further between shots
  - Score ≥30 → hoop moves on both axes between shots
  The platform's motion is purely vertical at every score tier.
  Bot doesn't currently handle the moving hoop. See "Open problems".
- **Reward thresholds** (gem-payout Hoops): trophy at a single-game
  score ≥40; Hoops pet at combined-3-try score ≥66; per-tier gem
  values not on any public page I could reach.
- **Variant note**: there's a newer **Swishy Hoops** (W7-era,
  [SteamDB patch notes](https://steamdb.info/patchnotes/20697793/))
  that uses **3 lives** per session and gives Ball Points instead of
  gems. The bot's `lives_diff` machinery suggests we're already
  targeting Swishy Hoops; the daily-3 rule may not apply there.
  Worth verifying in-game.
- **Extra plays per day**: the Gem Shop's "Daily Minigame Plays"
  upgrade reportedly applies to Catching, not Hoops. No Hoops-specific
  bribe / talent / shop entry surfaced in search. Unconfirmed.

## Game-physics assumptions (empirical)

- **Launch is fixed-power and fixed-direction.** Same platform Y at click +
  same hoop position → same trajectory. Tuning is therefore deterministic
  per shot: for any given hoop, there's exactly one optimal `platform_y`.
- **Click position has no measurable effect** on aim (verified May 3 with
  the click_sweep and click_extreme experiments — both produced ball
  trajectories within the ~20px noise floor regardless of click_x or
  click_y). The bot clicks at the hoop position because that's a sane
  default; it could click anywhere.
- **Hoop respawns only on a make**, at a random new (X, Y) for score
  0–19. Misses don't move it. This means every miss at the current
  hoop is "free" data (we'd be sitting on the same target anyway), and
  any strategy that *reduces* shot frequency is counterproductive —
  the bot has to fire something to make progress. Build the predictor
  by firing and logging, not by holding back when uncertain.
- **At score ≥10**, the hoop starts moving horizontally between shots
  (corrected 2026-05-23: the wiki claimed the platform was what moved;
  live observation showed otherwise). The platform stays stationary in x
  at every score tier. The `home_x` / `X_TOLERANCE` machinery was built
  for the wrong subject and is disabled (tolerance=9999); left in place
  as machinery in case score-30+ surprises us.
- **At score ≥20**, the hoop motion gets more aggressive. Bot reads
  hoop position once per shot when target_y is None, so by fire time
  the position can be stale. Not handled yet.
- **At score ≥30**, the hoop moves on both axes. Not handled.

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

1. **Score ≥10 moving hoop** — bot reads hoop position once when
   `target_y` is None, then spends ~seconds waiting for the platform
   to reach target_y. By fire time the hoop has drifted. Fix is to
   re-detect hoop position right before firing (and recompute target_y
   off the fresh position) rather than caching the first detection.
2. **Score ≥30 two-axis hoop motion** — same fix as above probably
   covers it, since the re-detection picks up both axes. Verify after
   the score-10 fix lands.
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
