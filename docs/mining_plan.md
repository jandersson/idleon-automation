# Mining bot plan — closing #5 (click policy) and #6 (score/digits)

Working plan for finishing the mining bot. Companion to the historical
`mining_backlog.md` (visual/mechanical context) and issues #5/#6. Written
2026-06-15.

## The mechanic (grounded in the game, not guessed)

Sources: idleon.wiki/wiki/Mining, digitaltq.com mining guide, Steam
discussions. The cart minigame is an endless runner:

- **Two actions, one button.** Click = the cart **jumps** up. Click again
  while airborne = the cart **slams** down (fast descent).
- **Scoring is the slam onto ore.** Landing a slam on an ore vein scores:
  **1** (copper), **2**, **3** (yellow/starfire) — by ore colour. A slam
  onto ore also **rebounds the cart upward** (a free jump), which sets up
  clearing the obstacle after it.
- **Pits kill.** Falling into a dark gap ends the run. Jump to clear them.
- **Speed ramps** over the run; the rhythm tightens. Dev highscore is 103.
- **5 attempts/day**, pooled with chopping/catching — data is scarce, so
  runs must be instrumented to extract maximum signal per attempt.

Canonical rhythm from the guides: *slam on the ore, then immediately jump
again right before the next pit.* The bot needs both halves — jump-to-clear
(survival) and jump→slam-to-score (points) — and a state machine that
knows whether the cart is grounded or airborne.

Coordinate facts already settled (see `mining_backlog.md`): the cart
sprite is **fixed on screen**; the track scrolls left at **~93 px/s**.
"Ore under the cart" therefore means `find_next_terrain` distance → ~0.
Cart altitude is readable from `cart_y` vs `plank_y` (the cart sprite
rises on a jump), and a slam pose template exists
(`assets/cart_slam_small.png`).

## Phase 0 — instrumentation (done 2026-06-15)

Prereq so live runs are productive and don't waste attempts on a blind
bot:

- **`mining --watch`** (observe mode): the bot does *not* click Play Game
  and does *not* auto-jump. It runs the detector, logs every human click
  to `mining.db` with the detector state at fire time (`source='human'`),
  captures digit crops, and prints live telemetry. Lets a human play a
  good attempt while the bot records ground truth for the slam policy.
- **Live digit capture** (`digit_capture.DigitCapturer`): every distinct
  PTS crop seen during any run is saved to
  `assets/digit_capture/<stamp>/` (deduped by binarized pixels → one PNG
  per rendered number). Runs in both `--watch` and auto mode. This is the
  digit-bootstrap vehicle the offline `bootstrap_digit_templates.py` can't
  provide for mining (no fixed score region, no saved per-jump frames).
- **Click-timing fix**: the auto-jump now fires the click *before* the
  `log_jump` DB insert (issue #5's explicit requirement; CLAUDE.md "Click
  timing"). The plank-x-range compute moved after the click too — it runs
  on the already-captured frame, so the value is identical.

## Phase A — collect data via live runs

Order matters because attempts are scarce:

1. **One `--watch` run, human plays well.** Yields simultaneously:
   (a) digit crops climbing through 1-9 for the OCR bootstrap;
   (b) human click timing vs detector state (cart_y/airborne, ore/pit
   distance) — the design data for slam timing, the darts-from-`centroid_dy`
   playbook; (c) end-to-end detector validation (does cart/pit/ore/plank/
   score track live?).
2. **A few auto runs, jump-only.** Populate `jumps` with
   `next_distance_px` + outcome so `survival_rate_by_distance()` has rows
   (currently 0). Confirms the trigger window actually fires.

After each run: read the session log + query `mining.db` + view the
captured crops directly — don't ask for paste-backs.

## Phase B — #6: digit templates + multi-digit box

- Label the captured crops (they render as images; the score is monotonic
  over a run, which makes labels self-checking). Feed per-digit components
  to `score_template_ocr.save_template` → `digit_templates/1.png..9.png`.
  Add a mining branch / labels-file path to
  `scripts/bootstrap_digit_templates.py`.
- Validate `read_score` on 2- and 3-digit scores. The box (`PTS_DX`,
  `PTS_DY` in `detector.py`) was derived from a single-digit "0"; if a
  leading digit is clipped, widen `PTS_DX` left. Add a regression test
  with a real multi-digit crop fixture.

## Phase C — #5 part 1: calibrate the jump trigger

- Build `survival_rate_by_distance()` from Phase A auto runs; set
  `JUMP_TRIGGER_MIN/MAX` to the band that actually survives.
- Ground it in physics, not just bins: from `--watch`/trace `cart_y`
  traces, measure jump-arc duration (frames airborne after a click) and
  confirm scroll speed (~93 px/s). Trigger distance ≈ scroll_speed ×
  time-from-click-to-airborne so the cart is up while the pit passes.

## Phase D — #5 part 2: slam-on-ore scoring (the unbuilt half)

- **State machine** keyed on cart altitude (`cart_y` vs `plank_y`,
  cross-checked with the slam-pose template):
  - grounded + pit in window → jump to clear;
  - grounded + ore approaching → jump to get airborne ahead of it;
  - airborne + ore ~under cart (distance → 0, lead-compensated for click
    latency + descent) → slam to score + rebound;
  - airborne + pit under cart → ride the arc (no slam).
- **Slam timing** is the empirical crux: learn the lead from Phase A human
  data (at what ore-distance / cart_y did good human slams fire?), then
  measure click→slam latency + descent speed to convert to a trigger.
- **Schema additions** (via `jump_log._LATE_COLUMNS`, so existing DBs
  migrate): `action` ('jump'|'slam'), `cart_airborne` (or cart_y−plank_y),
  `cart_pose`, `score_at_click`. Log every jump *and* slam with outcome
  (survived / scored Δpts / died) so the policy is tunable from the DB,
  per the per-bot data-tracking pattern.
- Enrich `find_cart` to surface the matched template name + width (pose),
  needed for `cart_pose` and airborne detection.

## Live-run log (2026-06-15)

Two attempts spent (watch + bot), both ended on pit 1, but both were
informative:

- **Detection + OCR validated live.** Plank/cart/pit tracked; the captured
  single-digit `0` matched the template (OCR read 0). Scroll ~94 px/s.
- **The loop ran at ~2 FPS — the real #5 blocker.** Fixed: cart tracking in
  a narrow column + dedup'd calls → ~40 FPS grounded, ~48 FPS through jumps
  (validated on traces; pixel-identical detection). See the
  `perf(mining)` / `fix(mining)` commits.
- **Jump arc measured (from the existing trace_20260516, no attempt spent):
  ~0.9s airborne, peak ~0.5s after the click, ~70px peak height.** This
  explained the bot's death: it fired at pit_dist=88, and the pit reached
  the cart at 88/94 ≈ 0.95s — exactly as the cart landed. Trigger lowered
  `[40,90] → [30,50]` so the pit arrives at the cart near the arc's peak.
- **Data-quality fixes:** the human-click listener was double-logging the
  bot's own clicks (now watch-only); the fatal-jump outcome logged
  'unknown' (now cart-gone → 'died').

**Next:** one bot run at the new `[30,50]` trigger to validate survival.
The `[traj]` log lines now print the cart's per-frame arc at the live
resolution, so that run confirms the 0.9s arc and pins the trigger. If the
cart still lands in the pit, the arc trace says whether to fire earlier or
later. Then repeat for a survival-by-distance curve. Digits 1-9 (#6) still
need a scoring run (human, or the bot once slam is built).

## Open questions

- Does a slam onto an ore reliably rebound enough to clear the *next* pit,
  or is a fresh jump still needed? (Observe in Phase A.)
- Ore colour → point value: worth prioritising high-value ore, or just
  slam every ore? (Decide after seeing point density.)
- Two-click jump→slam combo latency: is `random_delay`'s 80-200ms between
  the two clicks tight enough at high speed, or does the slam need a
  shorter, fixed inter-click gap?
