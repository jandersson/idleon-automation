# Choppin minigame — mechanics + bot architecture notes

Started 2026-06-11 as community-sourced mechanics research; rewritten
2026-07-11 after the planner era (runs 6-17) settled most questions
with instrumented data. History lives in issue #46's comment thread
and the chops/polls tables of `assets/chopping.db`; this file is the
current model. Best bot score: **52** (run 13); human best: 66.

## Mechanics (settled, instrumented)

- **Scoring: green +1, gold +2** — verified per-chop by the PTS OCR
  trail (every gold impact planned mid-zone pays +2; the 2026-07-06
  "gold pays +1 sometimes" era was entry-edge fires being evaluated
  off-zone, fixed by center-targeting). Score thresholds that matter:
  quest at 150, achievement at 141, mage skill scales with account
  best — peak score is the objective.
- **PTS counter updates instantly on a scoring chop** (maintainer
  ground truth) — but gold's +2 lands as TWO ticks, the second up to
  seconds late. Settled step-function values (polls.pts_read) are the
  analysis anchor; single reads 0.35s post-click under-read golds by 1.
- **The speed ramp is TIME-accumulated and applied at chop events.**
  A round idled ~15 min at a normal ~211 px/s, then the FIRST chop
  jumped the leaf to median 1684 / p90 2511 px/s (~10x). The original
  "chops ramp it, per-chop" finding was confounded (chops ∝ time at
  normal cadence); the ~600-650 px/s "saturation" is just f(t) at
  normal round lengths. Calibration points: +42s → +34 px/s;
  +15 min → +1500 px/s. **Operational rule: zero idle — the bot
  auto-clicks Play Game the moment it sees the prompt.**
- **Waiting mid-round is free at ≤45s scales** (pause experiments:
  flat speeds inside chop-free pauses) but see above — round AGE
  still accrues ramp that the next chop applies.
- **Deaths bank the points.** Clicking red ends the round; the
  accumulated points pay out the same as a voluntary exit. This makes
  late-round risk cheap (the EV ranking uses it).
- **No inactivity timeout** observed (≤45s tested); rounds end only
  by red-click or in-game exit.
- **Leaf motion is EASED**: x(θ) = (W/2)(1−cosθ). Mid-bar ~550-750
  px/s vs ~280-420 at the edges at typical speeds. All planning uses
  this model; its measured arrival error is +5ms mean / ~12ms sigma
  mid-bar at ≤120ms horizons, degrading superlinearly near the
  turnarounds (see the edge exponent below).
- **A registered chop re-rolls the zone layout within 86-201ms** (the
  in-game ack); the layout is otherwise STATIC between chops — a
  red-flanked gold stays unfireable until something chops.
- **Chop cooldown**: clicks ≤225ms after a registered chop are
  silently ignored; ≥588ms observed gaps all scored. (225, 588)ms
  unbisected; the fire hold (MIN_INTERCHOP_S=0.45 + re-roll ack)
  makes it moot at planner cadence.
- **The layout re-roll ack is necessary-but-not-sufficient** as a
  scoring signal (it false-acks on detection flicker); pts_read is
  the scoring truth.
- **The overlay anchors above the PLAYER** — its screen position
  changes per environment/map. Never cache overlay coordinates
  (regions.json is fallback-only); the bar is auto-located visually.
- "Gold slows the leaf" (community): no signal in any session.

## Bot architecture (the planner era, 2026-07-06 →)

`main.py` flow: auto-locate → auto-play → planned shots → banked exit.

- **Overlay auto-location** (`detector.find_bar_rect` +
  `derive_overlay_regions`): the bar is found as a wide thin band
  where zone green and red coexist (contiguous column runs, 1px gap
  bridge; decorative end caps split off); leaf strip / Chop button /
  PTS counter hang off it at offsets calibrated from the original
  hand-picked regions. The bot WAITS for the bar (~1/s) instead of
  exiting, so it can be started before the round.
- **Auto-start** (`detector.find_play_button`): while waiting, the
  'Play Game' prompt is template-matched (shared sprite with mining;
  0.992 live match) and clicked — capped 2/session (each consumes a
  shared daily try). Zero idle by construction.
- **Planned shots** (`planner.py`): pick a target impact inside an
  upcoming zone crossing, schedule the click for impact −
  CLICK_LATENCY_S (55ms), re-plan every poll, commit inside an 80ms
  window with a sleep/spin release (sub-ms). Fires from any leaf
  position; impacts land mid-zone. Every zone yields a sweep-0 and a
  sweep-1 (through-bounce) candidate; red boundary CROSSING TIMES
  from both sweeps bound every margin (turnaround pockets repel from
  both sides). Cross-sweep candidates steer ~36% of approaches
  (plan_saw_sweep1) but re-label to sweep-0 by commit time.
- **Error model (all terms measured)**: sigma_local = 12ms (mid-bar
  fire residuals) × (1/sinθ)^1.5 at the target (thin EDGE fires died
  at 29% vs mid 8% under linear scaling — five-death split) + 5% ×
  horizon (V_max estimator drift). Hard feasibility rungs by drought:
  3.0σ fresh / 2.5σ at 15s / 2.25σ at 30s.
- **EV ranking**: candidates rank by value − death_cost ×
  death_risk(σ-margin); death_risk is the empirical fit exp(1.63 −
  1.39k) (P(3σ)=0.08 — the real tail is ~50x fatter than Gaussian at
  the floor), death_cost is the remaining round's points decaying
  with age and drought. A floor gold loses to a fat green while the
  round is worth living for. EV reorders but NEVER refuses: a lone
  rung-passing candidate fires (the alternative on a static layout is
  the starve exit — same banked points, zero upside).
- **Gold chase** (`gold_wait.py`): same-sweep ride + cross-pass wait
  share a 6s give-up clock and a viability pricing check
  (`detector.gold_window_ms` — don't chase a gold whose best entry
  window can't clear the gate at current speed).
- **Starve exit at 40s** (was 60): the longest drought that ever
  revived in the planner era is 31.2s; nothing in (32, 60)s ever
  fired again. Banks and tells the user to exit in-game.
- **PTS OCR**: template-based (`common.score_template_ocr`), digit
  library complete (0-9) in `assets/digit_templates/` — tesseract is
  unusable on the pixel font. Suffix-tolerant reader (the "PTS"
  label), binarize threshold 200 (white-on-sky), plausibility gate
  (monotonic, ≤+2/chop). Continuous ~4Hz sampling on non-fireable
  polls → polls.pts_read step function = scoring ground truth.
- **Instrumentation**: every fire in chops (plan fields: aim_mode,
  target_x, plan_margin_ms, plan_impact_in_ms, plan_sweep,
  plan_saw_sweep1, gold_wait_ms), full-rate polls with layout RLE +
  pts_read; --save-frames (launcher toggle) persists leaf+bar
  composites (~2Hz + per chop) and score crops. DB commits every ~2s
  so aborts lose nothing.
- **Legacy fallback**: CHOPPING_AIM=gate (the reactive time-to-red
  gate that starved at saturation), CHOPPING_AUTOREGIONS=off
  (regions.json). Kept for A/B only.

## Post-mortem discipline

Every death gets replayed (`plan_shot` on the logged state — pass
sigma_ms=12, the live value; the signature default is 16 and silently
rejects everything the bot fired). Deaths so far: 2 pre-EV avoidable
floor-golds (led to EV ranking), 3 vindicated forced fires (lone
candidates / drought rungs on hostile layouts), of which the edge
cluster led to the ^1.5 exponent. The maintainer's direct in-game
observation outranks any statistical reconstruction — two plausible
theories died in one evening against one sentence each (see memory:
verify-mechanics-with-user).

## Open questions

- The time-ramp function f(round_age): only two calibration points.
  More large-idle observations would pin the shape (observational —
  don't burn tries on it).
- Gold-zone spawn triggers (observational; gold sits on the bar in
  ~95% of long-session polls).
- The (225, 588)ms chop-cooldown boundary (moot at current cadence).
- #110 hit-point offset: absorbed by center-targeting; the offset
  number itself is recoverable from pts_read × fire depth if ever
  wanted.

## Sources

- IdleOn wiki — Choppin: <https://idleon.wiki/wiki/Choppin> (403 to
  WebFetch; use curl / the MediaWiki API)
- DigitalTQ Choppin guide: <https://www.digitaltq.com/wiki/idleon/choppin-guide>
- Steam: "Tips for minigames", "The chopping and mining mini-game are
  not fun or completely broken" (see git history for URLs)
- Everything else: `assets/chopping.db` + issue #46's comment thread.
