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

### Run 3-4 (bot, [30,50] trigger, then --save-frames)

- **The `[30,50]` trigger clears the first pit** (reproduced twice). The
  bot fires at ~50, the cart arcs over the pit and lands on solid plank.
  Physics-based trigger validated for an isolated pit. Note a ~0.2s
  click→launch delay (the cart is still grounded at dt+0.16).
- **It then dies ~1.9s later on a closely-following obstacle**, not on
  pit 1. From the saved frames (`--save-frames`, botrun_20260615_012340):
  after clearing pit 1 the cart lands, and a second hazard arrives almost
  immediately — a second pit and/or the orange obstacle. The single
  jump-per-pit policy can't sustain the rhythm.
- **Visual ground truth (finally have full frames):** pits are dark spiked
  gaps in the plank; **ore is the silver-blue crystal piles sitting ABOVE
  the plank** (slam those to score — the wiki was right, the backlog's
  "copper/orange" note was wrong); there's also an **orange-brown obstacle
  on the plank surface** that the tan-hued ore scanner picks up as "ore"
  and that the cart appears to die on. Needs classification (trap? must be
  jumped?).
- **Airborne cart detection is brittle.** The cart goes undetected during
  the jump (cart=None, FPS drops): the sprite barely changes but the
  BACKGROUND does (cave wall + ore behind the airborne cart vs plank
  behind the grounded one), so TM_CCOEFF_NORMED drops below 0.80
  (airborne best ~0.75-0.79). Lowering the threshold admits false matches
  (a cart_large@0.5 ghost at the grounded row) — rejected. Added one
  live-res jump-pose template (`cart_jump.png`, from frame 35) — it nails
  that pose (1.0) with no false positives, but the pose+background vary
  through the arc so one template only covers part of it. Robust airborne
  detection needs several jump-pose templates or a background-masked /
  color-based cart finder.

### Run 5 (offline, no attempt spent) — airborne detection fixed

Task 1 below was the blocker; it's resolved **without spending an
attempt**, against the existing `botrun_20260615_012340` frames.

- **Cart matching is now background-masked.** Root cause of the mid-jump
  blindness: the unmasked `TM_CCOEFF_NORMED` includes the background baked
  into the template, and during a jump that background scrolls from plank
  to cave-wall+ore, dropping the score to ~0.75-0.79 (below 0.80) for the
  launch/peak/pre-land poses. Fix: `common.templates.match_multiscale_masked`
  (`TM_CCORR_NORMED` over a cart-silhouette mask = plank-tan pixels
  excluded), wired into `detector._match_cart`. A **single grounded
  `cart_small` template now generalizes to the entire arc** — no
  pose-specific templates needed. Validated on all 61 frames: grounded
  0.99, airborne arc (frames 34-42) **0.86-1.00, every frame detected**
  (was NONE at 34/36/37/41/42). Threshold raised 0.80 → **0.85** (masked
  CCORR scores higher; 0.85 clears the worst real airborne frame at 0.86
  and rejects the ~0.81 game-over-screen floor — which is anyway gated
  out because `_find_plank_top_y` returns None on the game-over screen).
- **Methodology note:** extracting per-frame jump templates from this one
  capture and testing on it would be circular (memorizing one arc). The
  masked approach is non-circular — the template comes from the *grounded*
  frames and generalizes to *airborne* poses purely because the mask
  removes the only thing that differs (background).
- **Scale 0.5 dropped + tracking column tightened 90 → 50px.** Small-scale
  masked templates score ~0.89 on off-column background; no real pose needs
  0.5 (slam=0.75, peak=1.0). Mid-arc frames (35-41) track at the true
  x≈150; only the launch (34) and dying pre-pit (42) frames are ~30px
  approximate, and the bot can't re-fire mid-jump so those don't reach a
  decision.
- **`cart_jump.png` is now redundant** (masked `cart_small` covers the peak
  too) but harmless — left in place; the masked match still picks it when
  it scores highest (frame 35: 1.00).
- Tests: `test_templates.py` gains masked-match + background-independence
  guards; `test_mining_detector.py` cart frames now textured (masked CCORR
  is degenerate on perfectly-flat synthetic frames, an artifact real cave
  frames never hit). 408 green.

### Run 6 (offline, no attempt spent) — "orange obstacle" + ore, resolved

Investigated the task-2 questions against the same `botrun_20260615_012340`
frames. Both prior assumptions were wrong; the net result is a detector
correction, not a new feature.

- **There is no distinct "orange obstacle."** HSV-segmenting the plank band
  shows the only tan signal above the plank is the *continuous cave wall*
  (the boulder backdrop), which the old tan-hue `_scan_plank_ore` was
  firing on. The lethal hazard is the **pit** (dark gap in the plank
  surface), already detected by the pit scan. The prior session's "orange
  trap" was this cave-wall false-positive.
- **The silver-blue crystal piles are static parallax BACKGROUND, not
  slam-ore.** Per-frame position proves it: across every grounded frame the
  blue piles hold fixed screen x (91 / 196 / 301 / 406) while the
  foreground pit scrolls left smoothly at ~3 px/frame (306→258 over frames
  7–20). Static-every-frame rules out aliasing. So the blue is decoration
  the cart can't interact with — and a naive "ore = blue" scan is *worse*
  than the tan one.
- **Phantom ore actively breaks survival.** `find_next_terrain` returns the
  nearest pit-OR-ore and the jump trigger fires only on `kind=='pit'`
  (`main.py`). A blue-ore scan reported "ore at distance 28" every frame
  from the static x=196 pile — which sits between the cart and the real
  pit and never scrolls away, so it would mask **every** pit and guarantee
  a death. (Verified: with ore disabled, `find_next_terrain` cleanly tracks
  the real pit, distance 138→84 as it approaches the [30,50] trigger.)
- **Decision: `_scan_plank_ore` returns [] (ore detection disabled).**
  Detecting nothing is survival-correct until the real ore is known;
  the function/constants/x-range plumbing are preserved as scaffold.
- **The real lesson — foreground vs background.** The detector has no
  foreground/background separation: the pit scan also picks up a static
  background gap at x=138 (harmless only because it sits behind the cart).
  The clean key is scroll velocity — foreground ≈93 px/s, background
  static. The real slam-ore (which never appeared in this 0-score capture)
  should be characterised from a **scoring `--watch` run** and separated
  from decoration by that velocity filter.
- Tests: the three tan-ore detection tests collapse into one asserting the
  disabled-[] contract (flip it together with the function when real ore
  lands). 408 green.

### Next session

1. ~~**Robust airborne cart detection**~~ — **DONE (Run 5).** Continuous
   through a jump via background-masked matching.
2. ~~**Identify the orange obstacle / separate ore from hazard**~~ —
   **DONE (Run 6):** no orange obstacle (cave wall); blue piles are
   background, not ore; ore detection disabled as phantom. Real ore +
   foreground/background velocity separation deferred to a scoring run.
3. **Spaced-obstacle / multi-jump policy** — the current death (a second
   obstacle right after the first jump). Detection is now continuous, so
   this is unblocked. Likely needs a shorter cooldown + reacting the
   instant the cart lands. **Wants a live `--save-frames` run** capturing
   a longer multi-jump sequence (detection is no longer the bottleneck).
   Worth adding scroll-velocity tracking here — it both filters background
   pits and is the prerequisite for real ore detection.
4. **Then** slam-on-ore scoring (Phase D) and the digit bootstrap (#6),
   both of which need a *scoring* run to gather data.

Trigger `[30,50]` is good for isolated pits — don't re-tune it without
cause; the deaths are now downstream of it.

## Open questions

- Does a slam onto an ore reliably rebound enough to clear the *next* pit,
  or is a fresh jump still needed? (Observe in Phase A.)
- Ore colour → point value: worth prioritising high-value ore, or just
  slam every ore? (Decide after seeing point density.)
- Two-click jump→slam combo latency: is `random_delay`'s 80-200ms between
  the two clicks tight enough at high speed, or does the slam need a
  shorter, fixed inter-click gap?
