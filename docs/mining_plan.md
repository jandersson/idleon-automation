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

### Run 7 (offline, no attempt spent) — multi-agent audit + fix sweep

Ran a 7-agent audit workflow over the whole mining subsystem, then
implemented every offline-fixable finding that could be validated against
existing frames / pure logic / schema. Six commits (`b03b2f0`→`5817a45`).
The headline finding: **the spaced-obstacle death has a concrete mechanism**
— a click while the cart is airborne is a SLAM (fast descent), and
`JUMP_COOLDOWN_S=0.6s` expired ~0.3s before the ~0.9s arc landed, so a
second pit in `[30,50]` mid-arc fired a lethal slam into the pit.

Shipped (all with tests, 429 green):

- **Airborne guard** (`policy.py`, the death fix): `should_jump()` refuses to
  fire while the cart is airborne, judged by `cart_y` vs a rolling-MAX
  grounded baseline (max never lets airborne dips cause a false-airborne
  suppression of a real jump; baseline=None until warmed → defaults to
  grounded). Validated on the botrun arc (peak frames classify airborne).
- **find_next_terrain hardening:** `_find_plank_x_range` now bridges a
  wide-pit plank split (returns the full extent, not just the densest
  cluster) so an incoming pit past a >100px split is still scanned; scan_x
  clamped to the plank left edge; `plank_range` threaded from the loop
  (drops a redundant HSV+cluster pass).
- **`_extract_runs` merge guard** — only bridges a gap when one run already
  meets `min_width`. This **also eliminated the x=138 "static background
  pit"** the Run-6 note flagged: re-validation showed it was never a real
  gap, just two isolated 1px dark specks the old merge logic glued into a
  fake 6px pit. (So the velocity-filter's motivating example was a noise
  artifact — see deferred below.)
- **Dropped cart scale 0.6** — it ghosted the launch frame at the wrong x;
  dropping it snaps frame 34 to the true x. Canary test pins the Run-5
  match params.
- **Phase-D schema hooks + honest accounting:** `action`, `cart_height_
  above_plank`, `cart_pose`, `score_at_click` columns (logged at the bot +
  human sites); `survival_rate_by_distance(source='bot')` excluding
  human/legacy/pending rows; run-end settle relabels only the last pending
  jump 'died'; DB connection owned by `run()` (closed in finally).

Deferred, with reasons (NOT done — would be speculative or need live data):

- **Scroll-velocity foreground/background filter (`TerrainTracker`).** Built
  and unit-tested, then **removed**: its one concrete motivation (the x=138
  static pit) turned out to be the `_extract_runs` merge artifact above, and
  no real static-background pit remains in the data to validate against.
  Building a stateful tracker against a hypothetical violates "no speculative
  features". It's the prerequisite for real ore detection (also needs live
  data), so build it *then*, validated on a real static-vs-scrolling capture.
- **PTS OCR "P"-contamination fix.** The box right edge (`plank_x0+2`) catches
  the 'P' of "PTS" on some crops (a 2px stub → an unmatched component →
  read fails). But: (a) the bot's `final_score=0` is **correct** (it dies at
  0 points; clean frames *do* read 0), so OCR isn't blocking anything now;
  (b) the fix is entangled with `plank_x0` jitter and the units digit sits
  only ~2px from the 'P', so no fixed right-edge cut cleanly separates them —
  the precise box needs a real multi-digit crop to calibrate. Deferred to the
  live scoring `--watch` run; the safe form is a component guard validated
  against real ≥2-digit crops, not a blind geometry move.
- **`_find_plank_top_y` top-edge rewrite.** `plank_y` is rock-stable (200)
  while grounded and only jitters during the jump (when the airborne guard
  suppresses anyway), so the rewrite's benefit is marginal and it would force
  recalibrating every `*_SCAN_DY` offset. Skipped.
- **Prune cart templates to cart_small.** The extra templates exist for
  different window *resolutions* (cart_large covers windows cart_small×1.5
  can't reach); pruning is only validated at the botrun resolution, so it's a
  perf win with a cross-resolution correctness risk. Skipped (bot is 30+fps).

### Run 8 (live human --watch, 1 attempt) — first real play; detection bug + digits

First successful human watch run (`botrun_20260615_030254`, 17s, 10 clicks
logged, human scored 4). Two outcomes:

- **Digit bootstrap (#6), partially done.** The score climbed 0→1→2→3→4, so
  the live capture caught clean glyphs; `digit_templates/{1,2,3,4}.png`
  saved (visually confirmed; `read_score` now reads 0-4 and composed
  multi-digit numbers). 5-9 await a higher-scoring run. (The run logged
  `final_score=0` only because 1-4 had no templates *during* the run.)
- **A cart-tracking bug, found and fixed (`cfbfa69`).** After jump 1 the
  detector locked onto a false 0.85 match at x=631 — off-plank cave wall +
  a hanging chain — and stuck there the whole run while the real cart sat at
  x=150 (0.99). Mechanism: a transitional jump pose dipped the real cart
  below 0.85 in-column for one frame, the full-frame fallback grabbed the
  cave/chain, it became the prior, and the column-tracker re-found that
  stable false match every frame after. Fix: the cart x is fixed per run
  (issue #1), so `find_cart_detailed` rejects a fallback match that
  teleports >`CART_MAX_X_JUMP_PX` from the prior, and the loop keeps the
  prior anchored on a transient miss (re-acquires only after
  `CART_REACQUIRE_MISSES`). Replaying the 337 frames: 260/313 false-lock
  frames → 0.
- **Slam-timing data is contaminated.** Because the cart state was wrong
  (x=631) for jumps 2-10, those rows can't characterise slam lead/latency.
  The ~4 scoring slams happened but their cart_y / ore-distance weren't
  captured correctly. **Needs a fresh watch run on the fixed detector.**
- **Residual (smaller follow-up):** with the false-lock gone the detected
  cart_x still wanders ~108-263 during jump poses (the masked match lands a
  few px off-centre and the column follows). It's within the plank (not the
  cave) and mostly on airborne frames where the fire policy is suppressed,
  so it's not run-breaking — but firming the x to a stable established lock
  would clean up the terrain-scan origin during jumps.

### Run 9 (live human --watch ×2) — second detection bug fixed; clean data confirmed

Two more human watch runs, both initially broken by detection bugs that the
saved frames let us find + fix, the second then confirming a clean pipeline:

- **Run 9a (`botrun_20260615_031952`), a different world room.** The minigame
  overlay floats above the player, so a different room moved the layout:
  `_find_plank_top_y` picked a competing overworld-floor tan band at y=297
  over the real plank at y=190 (global argmax takes the brightest band), so
  terrain + score broke (next=None, score=None everywhere). Fixed (`99b7e34`)
  by anchoring the plank search to just below the cart's grounded y
  (`near_y` = `grounded.baseline()`). Re-running the frames: plank 297→190,
  terrain 197/220 (was 0), score read 0→1→2.
- **Run 9b (`botrun_20260615_032948`), same room, fixed build — CLEAN.** Cart
  stable at x=322 (no false-lock), `cart_height` sensible (6/13 grounded, 54
  airborne), terrain tracked (pit @16/122/25), score 0→1. Click 3 is a real
  human slam logged with its altitude (54px) + pit distance (25). Short run
  (scored 1, died at 4 clicks) so the slam sample is small, but every logged
  column is correct — the instrumentation + detector now produce clean
  Phase-D data. Both this-session detection bugs (cart false-lock `cfbfa69`,
  plank mis-detect `99b7e34`) are closed.

Net for the live-data goal: digit templates 0-4 bootstrapped (#6); detection
hardened across positions/rooms; clean slam-timing capture validated (needs a
longer run to accumulate enough slams to fit a Phase-D slam trigger).

### Run 10 (live AUTO bot, last attempt) — DIED IN PIT 1 (single jump didn't clear it)

First full auto run on the hardened detector (log `session_20260615_033359`,
no `--save-frames`). Outcome: 1 jump, **died at the first pit** (user watched
it directly). **Correction to an earlier misread of this log:** the `[traj]`
trace was first read as "cleared pit 1, died on a second obstacle" — that was
WRONG. The tell: after the jump, `pit_dist` froze at ~211 (211→212→210→208)
instead of decreasing, which means the **scroll had stopped = the run already
ended**. The cart's arc came back down INTO the first pit (~dt 1.16) and sank;
the "second pit at 211" was a static feature on the frozen death screen. There
was no second obstacle.

What the trace does show (still valid):
- Detection held live: cold-start plank_y=297 for 1 frame, then anchored to
  ~185 (room fix worked); cart stable at x=322; terrain tracked.
- Airborne guard fired correctly: `airborne=True` across the arc, no spurious
  mid-arc click.
- Jump fired at pit_dist 49, cart rose to h≈76 — high enough — but the gap
  did NOT fully scroll past during the airborne window, so the cart landed
  back in it.

So the gating problem is more basic than "multi-jump": **a single jump at the
current trigger does not clear the first pit in this room.** The "clears pit 1"
claim in Runs 3–4 (a different room, cart_x=150) is now suspect — it may have
been the same misread, or genuinely room-dependent.

HYPOTHESIS (unverified — needs frames): the cart is airborne ~0.85s; the gap's
near edge arrived ~mid-arc (dt 0.53), leaving only ~0.55s ≈ ~52px of gap able
to pass under before landing. If the pit is wider than ~52px, no single jump
at this timing clears it — and firing EARLIER (larger trigger distance, gap
arrives just after launch) would give the gap the full ~0.85s / ~80px to pass.
This INVERTS the Run-3/4 reasoning that lowered [40,90]→[30,50]. Do NOT act on
this without a `--save-frames` capture: measure pit width + the airborne span,
then set the trigger so the whole gap passes while airborne. Daily attempts
exhausted; next session needs a frames run of the first-pit death.

### Run 11 analysis (2026-06-16) — the "fire earlier" hypothesis is BACKWARDS; fire LATER

Re-derived from the same death trace (session_20260616_131320, an exact repro
at pit_dist=49) in ABSOLUTE time, cross-checked by a 3-lens agent panel. The
gap reaches the cart at a fixed world-time regardless of when we jump; the
click only shifts the (fixed-length) airborne window. Survival needs the window
`[launch, land]` to contain the gap passage `[near, far]`. With v≈79 px/s,
click→launch δ≈0.23 s, airborne A≈0.88 s, this gives two bounds on the fire
distance D:

- **(1)** airborne before the gap arrives: `D ≥ δ·v ≈ 18`
- **(2)** gap clears before landing: `D + W ≤ v·(δ+A) ≈ 88`  (W = gap width)

The old `[30,50]` fired at ~49: `49 + 52 = 101 > 88`, so the cart landed ~13 px
into the far edge — exactly the death. Bound (2) shrinks with **smaller** D, so
the fix is to fire **LATER** (smaller trigger). Firing *earlier* (the Run-10
hypothesis) raises D and makes bound (2) worse — that hypothesis confused the
gap's arrival relative to the fire frame with its arrival relative to landing.

**Feasibility is still W-dependent (the real open question).** Point/center
collision with W≈52 → feasible band `D ∈ [18, 36]`, fire ~27. Full-footprint
collision (cart ~36 px, any overlap kills) → needs `W+F ≈ 88 px` of relative
travel but only `v·A ≈ 70 px` happens airborne → **no single-jump timing clears**
(short by ~19 px) and the fix is a multi-jump policy.

**Action taken:** trigger lowered to `[20,30]` (fires ~30, inside the point-model
band). This is the discriminator — a `--save-frames` re-run that **clears**
confirms the point model; one that **still dies, now landing at the far lip**
(not mid-gap), confirms footprint + W≈52 → build the land-and-re-jump policy.
Either outcome also lets us **measure the true W** from the frames. This is the
frames run Run 10 asked for, now with a corrected trigger to test.

### Run 12 (2026-06-16, `[20,30]` trigger) — pit 1 CLEARED; ore detection still blocked

`session_20260616_133056` (`--save-frames`, 130 frames): **JUMP #1 cleared pit
1 (survived)** — the `[20,30]` retune works, so a single jump DOES clear and the
point/center collision model holds (the footprint "impossible" branch is
falsified for this pit). The cart then crashed into the **ore** (user-observed),
which the bot can't see (ore detection disabled), so it jumped it like a pit and
sank.

**Ore characterisation attempted and FAILED — do not repeat as-is.** A 3-agent
panel reported the ore as a darker-tan band `V in [98,119]` "scrolling as
foreground." Re-checked directly against the frames: that tan is the **plank
surface** — with any workable column threshold it fires across the whole plank
on every frame (including before the ore arrives). The agent's "scrolling blob"
was the tan plank *between pits*, with the dark pit-gap scrolling through it.
Two other traps confirmed: (a) scroll-velocity can't isolate ore from plank
(both are foreground ~3px/frame; the velocity split only rejects the static
parallax crystals); (b) the saved frames are the FULL window (572×959) so
`_find_plank_top_y` mis-locks to a lower tan band (~296) — offline work must
crop to the play region or pass the live `plank_y~185`.

**What's actually needed (unchanged from the original Phase-A/D plan):** a
SCORING `--watch` run where a human jump-slams ore and the PTS counter
increments, with `--save-frames`. Isolate the ore from what sits under the cart
on the frame the score changes; measure the slam timing from that successful
slam (jump→2nd-click interval + the ore-under-cart position). Only then build
`_scan_plank_ore` + the slam policy (`should_slam`: airborne + ore under cart →
2nd click; never over a pit). Ore stays disabled until then.

### Run 13 (2026-06-16 scoring run) — ore CHARACTERISED; detection + slam BUILT

`botrun_20260616_135951` (`--watch --save-frames`): a human jump-slammed ore,
**PTS 0→1 at frame 132** — slam-on-ore scores, confirmed. Run 12's "no
separable signature" was the wrong band:
- **The ore is a brown rock pile that POKES UP above the plank-top line.** On
  the plank surface it's spectrally identical to the brown plank (both V~115),
  but in the band JUST ABOVE the top, bare plank is dark cave while an ore pile
  shows brown. Scanning that band past the cart returns discrete runs that
  scroll leftward across frames (validated f70→f159); bare plank / pits read
  zero. That above-plank brown is the discriminator (not surface colour, not
  scroll-velocity — plank and ore are both foreground).
- **Slam timing** from the human clicks (cart_y: grounded 179 vs airborne):
  jump at ore_dist~69 (grounded) → slam ~0.75s later at ore_dist~10 (airborne).

**Built (experimental):** `_scan_plank_ore` re-enabled (above-plank brown,
frac 0.25 / min-width 10); `should_jump` gains an ore window
`[ORE_JUMP_TRIGGER_MIN=40, MAX=70]` (farther than pits so the cart is airborne
when the ore arrives); new `should_slam` fires the 2nd click when airborne and
the nearest obstacle is ore within `ORE_SLAM_MAX_DIST=25` (self-timing off the
live ore distance, never over a pit). Regression-checked: ore stays empty on
the pit-1 approach and a nearer pit always wins, so pit-1 clearing is intact.
Tuned on ONE run/window — the next run's `kind='ore'` / `action='slam'` rows +
outcomes refine the trigger/slam distances and confirm the signature live.

### Run 14 (2026-06-16 auto) — detection good; slam was too EARLY

`botrun_20260616_152324`: JUMP #1 (pit, dist 29) survived, JUMP #2 (ore, dist
70) survived — **ore detection + ore-jump both work**. Then SLAM #3 fired at
ore_dist=17 and DIED: the cart came down in front of the ore and the ore hit
its side. Compared to the human scoring run — successful slams at ore_dist
**≤10**, human's own crash at 13, bot's crash at 17 — the slam was firing ~7px
too early. `ORE_SLAM_MAX_DIST` 25 → **11** (find_next_terrain floors ore_dist
at SCAN_BUFFER_PX=10, so 11 fires in the ~10-11 success zone). If the next run
shows the bot never slams (window too tight → runs into ore grounded), drop
SCAN_BUFFER_PX so the ore is seen closer and widen the window.

### Run 15 (2026-07-06, offline, no attempt spent) — rebound + speed ramp measured; baseline ratchet fixed

Replayed `botrun_20260616_135951` (the scoring run) through the live
pipeline end-to-end, emulating the loop state (grounded baseline anchoring,
cart prior) at the run's ~24.5 FPS cadence. Three findings, all acted on:

- **The slam rebound is a full free jump — and the existing policy already
  chains it.** Reconstructed arc: slam 1 clicked ~f113 (ore_dist 10),
  contact ~f118 at ore-top height, rebound to apex y≈97 (**82px above
  grounded 179 — same height as a grounded jump**, apex ~0.45s after
  contact); a second ore arrived mid-rebound and the human slammed it at
  f135 (dist 10–11, airborne) into a second full rebound. `should_slam`'s
  conditions (airborne + ore ≤ slam window + 0.5s cooldown) would have
  fired both slams — rebound chaining (#53 mechanic 2) needs no new code.
  Pits scrolling by mid-rebound are ridden out by the airborne guard as-is.
- **Scroll speed RAMPS within the run: ~82 px/s (first pit, f19-44) →
  ~86-89 mid-run → ~117 px/s (f147-161) — +40% in six seconds.** The
  trigger constants are distances but the physics bounds are times, so the
  static windows drift lethal as v grows (D ≥ δ·v needs 23px at v=100 vs
  static min 20). Built: `policy.ScrollVelocity` (px/s, wall-clock,
  gates against identity switches / death screens / plank mis-locks) +
  per-frame `scale_window`/`scale_slam_dist` by v/v_ref in the loop, with
  the static windows as the un-warmed fallback (#53 mechanic 3).
  `scroll_v_px_s` + `next_width_px` (the Run-11 gap width W) now log per
  jump, so W stops being frame archaeology.
- **GroundedBaseline decayed during the rebound chain and mis-anchored the
  plank.** The chain keeps the cart airborne longer than the 45-frame
  window; the window max decayed 179→159, the near_y plank search then
  locked a false tan band at y=149 (f141-146) and terrain froze at a
  spurious slam-range dist=11 — spurious slam fuel in auto mode. An
  eps-tolerant hold is defeated by slam-contact bounces (~20px above
  grounded) ratcheting the max down within eps; the level now never moves
  down except via a 200-sample safety cap (inflated-level recovery).
  Replay-validated: baseline holds 179 end-to-end, zero mis-lock frames.

Still live-gated: slam-window validation (the 25→11 retune has n=0 live
fires), ore VALUE discrimination (#53 mechanic 1 — every scored ore in the
captures is the same brown pile; platinum/starfire may not even match the
current brown HSV signature, i.e. would be INVISIBLE and crash the cart,
not just under-prioritised), and digits 5-9 for the PTS reader (#6).

### Runs 16-17 (2026-07-06 live) — FIRST BOT POINT; then a death that found an estimator bias

Two live auto runs on the Run-15 build, both attempts productive:

- **Run 16 (14:32): the bot's first scored point.** jump(pit, d=31, v≈92) →
  jump(ore, d=79) → slam(ore, d=11) — all survived, PTS 0→1, the slam
  retune validated (n=1) and the rebound rose to ~92px exactly as measured
  offline. It died later on a pit the scan lost right before arrival
  (width read collapsed 50→6 while plank_y flapped 184→192) — issue #103.
- **Run 17 (14:40, --save-frames): died at pit 1, and the frames convict
  the ScrollVelocity estimator.** True scroll ≈80 px/s (wall-clock
  anchors); the estimator reported 105.7, which scaled the pit window to
  [27,40] and fired at d=39. The scene froze with the pit at [427,483] and
  the cart at ~[449,485]: the far edge had passed the cart's RIGHT edge by
  2px but its center was 16px inside — landed at the far lip, the exact
  bound-(2) death. **Root cause: per-frame-pair velocity samples with a
  SCROLL_V_MIN floor.** Obstacle x is integer-quantized (~3px/frame), so
  pairs legitimately read 0px; the floor dropped only those (as "frozen
  scene") and kept the 6-7px pairs — a ~+30% systematic bias. Fixed by
  measuring velocity over an 8-frame (x, t) window (endpoint slope,
  ~0.3s), where zeros average in; the plausibility band applies to the
  window value only. Replay-validated on both captures: death run f32 reads
  82.3 (win [21,31] — d=39 correctly does NOT fire), scoring run tracks
  81 → 114 across the ramp with no overshoot.
- **Collision-model datum (the Run-11 open question):** the far edge
  passing cart_right is NOT sufficient — the reference is at/near the cart
  body, so bound (2) effectively needs ~half-cart extra margin. Don't
  re-derive the window top from physics constants; the empirical bins are
  the arbiter (D=29/31 survive at v≈80-92, D=39 dies at v≈80; A measured
  ~0.96s this run vs 0.88 earlier).

### Run 18 (2026-07-06 15:00 live, --save-frames) — chain reproduced; rebound-shadowed ore found a policy gap

Third live run of the day (PTS 1 again): jump(pit, d=29) → jump(ore, d=80)
→ slam(ore, d=12) all survived — the jump-slam rhythm is now reproducible
(slam n=2). The windowed ScrollVelocity fix validated live: 75-84 px/s
early, ~99-103 late, no overshoot; the pit jump fired at d=29 (correct
window, cleared cleanly).

The death found the next policy gap: **a rebound-shadowed ore.** The
slam's rebound arc (~1.1s airborne, apex 94px) kept the airborne guard
suppressing the ore-jump through the SECOND ore's entire scaled window
[46,80]; the cart landed with that ore at ~29px — below the ore window,
no branch fires — and the ore crashed into the grounded cart (frames:
freeze at f102 with the ore mashed against the cart's front; the Run-12
death mode, this time with detection working fine). Fix: an ore in the
PIT window now fires the pit-style late jump as a survival fallback — the
cart flies over, and mid-arc `should_slam` can convert the flyover into a
scoring slam for free (a low-altitude slam is untested territory; the DB's
`cart_height_above_plank` on slam rows will show how those land).

**Data caveat:** rows 56-57 (JUMP #4/#5, "pit d=25/28, died") are clicks
on the already-frozen death scene — their next_* context is garbage;
exclude them when reading pit survival bins (the scene froze ~0.3s before
JUMP #4, confirmed by the static full-width scans f102-127).

### Run 19 (2026-07-06 15:07 live, --save-frames) — slam 3-for-3; rebound-into-pit death (#104)

Fourth run of the day, PTS 1 again: jump(pit, 27) → jump(ore, 72) →
slam(ore, 11) — the chain is now routine (slam 3-for-3 since the ≤11
retune). Death: the slam's rebound descended into a pit that approached
during the arc (pit_dist 63→15 while airborne descending); the cart
touched down at the pit's lip with no legal move left (jump needs ~19px
of lead at v≈96, pit was at 10). The loss was determined mid-air — fixing
it needs pit-aware slam steering (slam early over bare plank to land
short and re-arm), filed as #104 with the blocking unknown "is a slam
onto bare plank a safe landing?".

**#104 unblocked same day:** the user verified slamming bare plank is a
safe landing. `should_steer_slam` shipped: airborne + nearest is a PIT in
the steer band `[50,75]` px at v_ref (scaled live) → slam onto the plank,
landing with the pit still in the grounded jump window. Below the band
both slam and ride land in the pit (don't fire, don't mislabel doomed
rows); above it the natural landing is safe (preserve scoring rebounds).
On run 19's death trace: pit first seen at 63 px inside the scaled band
[61,91] → steer slam → touchdown with pit at ~25 → jump window [25,38]
fires. Steer slams log as action='slam' with next_kind='pit' (scoring
slams carry next_kind='ore'), so the DB separates them for free. The
0.4s slam-descent estimate and the band edges are first-cut numbers —
the next runs' steer rows refine them.

Also fixed from this run's data: **outcome settling tolerated no
detection gaps** — row 59 logged 'died' because the cart was undetected
for the single frame its settle landed on, while it was alive mid-rebound
(the next action settled 'survived' moments later). `_settle_outcomes`
now keeps an absent-cart entry pending until the absence outlasts
CART_GONE_CONFIRM_S (0.6s); row 59 repaired to 'survived' in the DB.

### Run 20 (2026-07-06 15:19 live, --save-frames) — cleanest run yet; steer band bottom lowered

Fifth run of the day, PTS 1: jump(pit, 31) → jump(ore, 75) → slam(ore, 12,
scored) → jump(pit, 35, cleared a second pit mid-arc) — 4/4 survived
actions, slam 4-for-4, zero false steer fires despite #103 fragment
garbage during the rebound (readings at 10 and 145 both fell outside the
band, as designed).

The run still ended in a **two-pit spacing trap**: jump #4's arc landed
with the NEXT pit at ~10px (frozen death scene: grounded height 5, pit
static at 10). The steering slam almost had it — the second pit's first
sighting was 54px mid-descent, just under the scaled band bottom (63).
Two constants retuned from this trace:

- `STEER_SLAM_MIN` 50 → 40: the 0.4s descent cost behind 50 was measured
  from rebound apex (~90px); from jump-arc heights (~70px) a slam costs
  only ~25-30px of approach, so steering at ~50 lands jumpable.
- `JUMP_TRIGGER_MIN` 20 → 18: 18 IS the bound-(1) physics floor; the
  extra 2px only ever gated landed-short rescue jumps (post-steer
  touchdowns land with the pit at ~20-25 scaled), where it denied the
  rescue outright.

Both are marginal-rescue tuning — they convert this trap from certain
death into a tight-but-legal steer → land → jump sequence. Outcome-label
nuance: row 64 ("survived", d=35) is correct — its own pit was cleared;
the death belongs to the unactionable landing that followed.

### Run 21 (2026-07-06 15:24 live, --save-frames) — first live steer slam FIRED WRONG; plank lock + landing gate shipped

Sixth run of the day (PTS 1; jump/jump/slam chain fine, slam 5-for-5).
The first live steer slam fired — **and caused the death.** The frames
show a **three-pit gauntlet** (50px pits, 39px plank gaps): the ahead-only
scan reported a pit fragment 70px out, but the gauntlet's FIRST pit was
already directly under the airborne cart — invisible to a nearest-AHEAD
scan — and the slam dropped the cart straight into it (f86: cart
[449,485] inside pit [436,488]). Riding would have landed on the next
39px plank gap. Two fixes, both replay-validated on today's captures:

- **Per-run plank-row lock (`policy.PlankLock`, the #103 fix).** Two
  rival tan rows (185/190) flap the per-frame argmax; at the false row
  50px pits read as 7px fragments with wrong edges — which fed the steer
  its bogus 70px target and broke the 14:32 jump. Locks after 8 stable
  baseline-anchored frames (cold-start false bands can't lock — no
  baseline); `_find_plank_top_y(lock_y=...)` then pins the search to
  ±PLANK_LOCK_DY; releases after a 20-miss run (overlay gone). Replays:
  lock acquired ~f15, zero flap frames end-to-end, pit widths hold ~50
  through airborne stretches — the fatal fragment reading never happens.
- **Landing-solidity gate for the steer slam
  (`detector.plank_dark_fraction` over the cart's footprint).** The steer
  now requires an affirmative "wood below" (None/unknown rides): on BOTH
  the run-19 and run-21 descents the ground under the cart was already
  pit and the gate reads False — both would-be-fatal steers suppressed.
  The steer is now safe-or-silent; its positive case (wood below, pit
  ahead in band) hasn't occurred live yet.

Run 19's death class (rebound descending onto a pit already beneath it)
is thus NOT steerable mid-descent — the remaining lever is scoring-slam
timing (fire the ore slam earlier/later to move the rebound's landing
point), which stays deferred in #104.

### Run 22 (2026-07-06 15:35 live) — plank lock live debut clean; gauntlets are the wall (#105)

Seventh run of the day, PTS 1: the PlankLock held 184-187 all run (zero
flap frames live — #103's fix confirmed in production), pit widths read
full ~50px, all 4 actions survived (slam 6-for-6), and the steer slam was
correctly silent through the whole rebound (footprint-not-clear while the
next pit was in band; below the band once clear). No wrong decision
anywhere — and the run still died: jump #4 cleared its pit and landed
with the next pit at 10px. Third consecutive ~90px-gauntlet death. The
run-22 arithmetic (relative travel 124px/arc vs pit-N clearance capping
the landing ≤4px from pit N+1) shows single jumps CANNOT survive this
spacing at v≥95 — the options are a mid-arc slam into the 39px plank gap,
affine (not proportional) trigger tops, and/or two-obstacle lookahead.
Filed with the numbers as #105; that's the next session's design problem.

### Collision model — SETTLED (2026-07-06, #105 analysis; don't re-derive)

- **Pit collision is CENTER-based.** Discriminating measurement: run 22's
  jump #1 landed with its rear 12px over the jumped pit (center 6px clear
  of the far edge) and SURVIVED (`botrun_20260706_153531` f45-46, scene
  scrolling on); every captured death has the center inside a pit (+1px:
  run 22 jump #4; +5px: run 19; +16px: the 14:40 far-lip death; fully
  inside: run 21). The full-footprint model from the Run-11 caveat is
  falsified.
- **Consequences:** (a) bound (1)'s true floor is "airborne before the pit
  reaches the CENTER" = delta*v − half_cart(18) ≈ 3-6px, so landed-short
  states (pit at the 10px scan floor after an arc — the runs 20/22 deaths)
  are ESCAPABLE by an immediate jump: `JUMP_TRIGGER_MIN` dropped to 5
  (below the scan floor = any visible pit under the top fires). (b) a
  39px gauntlet gap gives the landing CENTER ±19px of tolerance, so
  gap-targeted firing is viable once the arc constant is pinned down.
- **Ore stays edge-based:** the run-18 death was ore contacting the
  grounded cart's FRONT, so ore jumps keep the delta*v floor
  (`ORE_FALLBACK_MIN`, clamped in should_jump's ore fallback).
- **Arc constant PINNED (same session):** the apparent ~15% disagreement
  was v-attribution error plus detection-lag inflation on survival
  landings (the world keeps scrolling between touchdown and the first
  grounded detection; death freezes don't). Re-measured with wall-clock
  [traj] velocities and freeze-exact clearances: **click→touchdown
  T ≈ 1.16-1.23s, speed-invariant**, and the clearance model
  `clear = v·T − dist − 18 − W` postdicts every exact landing within
  ~±5px — including run 22 jump #4's one-pixel death (fired at the
  v-scaled top after a ~5% v overread; model says clear 3±5).
- **Landing-targeted fire ceiling shipped (`policy.pit_fire_ceiling`):**
  pits now fire at `D* = v·1.2 − 18 − W − 12` (clamped [20,55]; falls
  back to the scaled top while the estimator warms). This aims the
  landing center ~12px past the far edge — alive-side margin 12±5, next
  gauntlet pit still rescue-jumpable (39−12 = 27 ≥ ~20 center-based
  launch need). Together with the rescue floor this is the gauntlet
  survival loop: targeted jump → land ~12 into the gap → rescue jump at
  ~15-27 → repeat.
- **Still open in #105:** two-obstacle lookahead (choose clear_target per
  gap when the next pit is visible at fire time), and live validation of
  the ceiling + rescue chain — T's calibration rests on n=3 exact
  landings, all from today's window/room.

### Run 23 (2026-07-06 16:03 live) — ceiling + rescue validated; slam altitude floor added

Deepest run yet (7 actions, 5 survived, PTS 1). The #105 machinery worked
live: jumps #4/#5 fired at 44/43 (the affine ceiling firing ABOVE the old
proportional top at v≈104-108) and landed cleanly with wide margins;
jump #1 survived a fire at 18 under the new floor; the post-jump-#4
landing had the next pit at 76 — the targeted landings are leaving room
instead of touching down at lips.

The death was a new interaction, user-observed as "the last jump slam
was a little too quick": JUMP #6 was the run-18 ore fallback (survival
flyover, ore at 33 below the v=122 ore window) and SLAM #7 fired ~0.15s
later at height ~22 — the first frame past the 18px airborne threshold,
with the ore still at the front bumper behind the 10px dist floor. The
slam dropped the cart BESIDE the ore (run-14/18 side-hit death). Fix:
`SLAM_MIN_HEIGHT_PX = 40` — every successful slam (6/6) fired from
69-76px, and a low flyover now stays a flyover (by 40px of height the
ore has scrolled under/behind, so the slam fires in proper geometry or
not at all).

### Run 24 (2026-07-06 16:07 live) — ore-shadowed rebound; steer slam extended to ore

PTS 1 (slam 7-for-7 — the altitude gate's first live pass at height 71).
Death: the rebound descended toward an ORE PAIR (dist ~90/~135) and
touched down with the first ore at the bumper (≤10px — below even the
ore-fallback floor), side-hit ~0.3s later; JUMP #4/#5 were clicks on the
frozen scene (rows 83-84, garbage context). Same geometry as the pit
version (#104) with an ore in the landing shadow.

Fix: `should_steer_slam` now accepts ORE in the steer band, gated to
REBOUND arcs only (`last_slam_time > last_click_time`) — steering early
lands with the ore at fallback-jumpable range and the run-18 flyover
takes it. Jump arcs stay pit-only: an in-band ore there is usually the
intended slam target of an ore-window jump, and steering would convert
scoring approaches into flyovers. No rebound scoring is sacrificed — an
ore that could still chain-score (slam range at height ≥40) is beyond
the steer band at steer time.

### Run 25 (2026-07-06 16:14 live) — PTS 2: double-slam chain + steer + rescue all fire and survive

First multi-point run, and every #104/#105 mechanism validated live in
one attempt: SLAM #3 (h71) scored → rebound to 88 → **SLAM #4 scored
mid-rebound** (the human-style chain, first time by the bot; scoring
slams now 8/9 lifetime — the one failure being run 23's too-quick slam,
since altitude-gated) → second rebound → **SLAM #5 = the first live
STEER slam** (pit at 92, in-band, solid below) → landed → JUMP #6 took
the steered pit → **JUMP #7 = the first successful rescue jump**
(fired at dist 10, rode over its pit). 8 actions, 6 survived.

The death: JUMP #7's forced rescue landed ~106px on, in the NEXT pit of
a dense field (below the steer band all arc — no branch can exist for
it; a rescue from 10 has no landing choice). Row 92 is a corpse click.
The remaining lever for dense fields is #105's two-obstacle lookahead:
choose landings while there is still a choice. #104 closed on this run.

### Run 26 (2026-07-06 16:17 live) — PTS 3; lookahead shipped; cadence issue found (#106)

New record: PTS 3 (triple slam chain — slams #3/#5/#8 all scored, plus
both steer variants firing and surviving, including the ore-steer's
first live outing). Death: the ore-steer's landing left ~0.28s to ore
contact vs ~0.3s click→launch — a photo finish lost because the ore was
FIRST SIGHTED at 73px, already deep in the band. The late sighting is a
loop-cadence problem: cart-miss frames (pose transitions) trigger
~0.25s full-frame template searches, blinding the bot ~27px per frame
at v≈109 exactly when sightings drive decisions → filed #106.

**Two-obstacle lookahead shipped (closes #105's remaining scope):**
`find_terrain_ahead` returns every obstacle ahead (nearest first;
find_next_terrain is now its head), the second obstacle logs per jump
(`next2_kind/x/width_px`), and `pit_fire_ceiling` takes the measured
gap to the next obstacle — tight gaps shrink the landing clear-target
(enter at the near edge, floor 4px) so the next obstacle stays out of
the forced-rescue zone; wide/unknown gaps keep the 12px default.

### #106 fixed (2026-07-06) — per-miss full-frame cart search removed

Benchmarked: the in-column search costs ~22ms, the full-frame fallback
~210ms, `--save-frames` imwrite ~5ms (not a factor). Every cart-miss
frame paid 22+210ms — the ~4fps stretches that delayed obstacle
sightings at arc transitions and caused run 26's photo-finish ore death.

The fallback is removed outright: the cart's x is fixed per run, so the
±50px column always contains the true cart — a miss is a transient pose
failure that self-heals in-column next frame, and the only thing a full
search could ADD is a false off-column match (the 2026-06-15 cave/chain
lock, now impossible by construction; CART_MAX_X_JUMP_PX deleted with
it). Genuinely-stale priors re-acquire via the existing
CART_REACQUIRE_MISSES path (prior=None → full search). Replay-validated
on three captures: miss streaks are single frames mid-run (6-10 only at
run-end despawn), zero mid-run losses — the fallback never contributed
a legitimate detection. Miss frames now cost ~22ms; the loop should
hold ~25fps through arcs.

### Runs 27-28 (2026-07-06 16:34 live) — #106 verified; residual death class isolated (#107)

**The #106 fix is visible in both logs:** [traj] ticks a steady 0.04s
through entire arcs (25fps where run 26 had 0.25s holes), the cart stays
tracked mid-arc (heights 127-133 where it used to read None at 4fps),
and sightings arrive a full band earlier. Run 27 also shows the first
gap-aware ceiling fire (JUMP #4 at d=50, v=106) and the first jump-arc
slam conversion under the altitude gate (SLAM #5 at height ~47, scored —
PTS 2). Run 28: PTS 1.

**Both deaths are the same residual class, now isolated as #107:** a
rebound landing pinned onto a sub-floor obstacle where the mid-air steer
window was empty — run 27 landed center-in-pit (pit below the band all
arc, ground below unsolid = that same pit); run 28 had an in-band ore
mid-rebound but the jumped pit was still under the footprint (solidity
gate correctly suppressed) and by footprint-clear the ore was below the
band. Every mechanism behaved correctly; the landing was fixed the
moment the scoring slam connected. The lever is one decision earlier —
choose WHERE in the ~34px ore window to fire the ore jump so the
predicted rebound landing clears obstacles (#107; needs the two-arc
travel constant measured from the 12 captured scoring slams).

### #107 first delivery (2026-07-06) — ore-jump fire placement

The chain model: SHIFT(click→rebound touchdown) = D_fire + 18 (pure
distance, click→slam contact) + v·T_REB (rebound leg). **Calibration is
coarse:** landmark measurement fails (everything visible at click
scrolls out before touchdown; occluded arc frames defeat cumulative
scroll counting), and log back-calculation pins T_REB only to
**0.9-1.3s** → ~±15px landing-prediction error against a ~±17px lever.
Shipped accordingly as a REPELLENT, not a targeting laser:
`policy.ore_fire_pick` scans the ore window from entry downward, keeps
the FIRST fire distance whose predicted worst-obstacle margin clears
ORE_PICK_SAFE_MARGIN_PX (15 — the error bar), and falls back to argmax
margin when nothing is safe. Window-entry behaviour is preserved
whenever it is safe; None (unwarmed v / nothing downstream) keeps the
default entirely. Consumes the full lookahead list (run 28's fatal ore
was the THIRD obstacle — next2 alone wouldn't have seen it).

Postdiction on the motivating deaths: run 28's pick would have fired at
the window bottom, landing with the fatal ore at ~20px instead of 0 —
marginal, but converts certain death into a fallback-jump race. Run
27's fatal rebound came from a JUMP-ARC CONVERSION slam (a pit-jump
chain — this lever doesn't apply); extending the same landing
prediction to the conversion-slam decision is the natural follow-up,
noted in #107. Refining T_REB needs runtime touchdown instrumentation
(the [traj] lines carry it; a settle-time landing log column would make
it queryable).

### Run 29 (2026-07-06 16:50 live) — PTS 4; 22 actions, 20 survived; the full stack works

New record on every axis: PTS 4, 22 actions (previous best 11), ~18s of
play, speeds reaching v≈130. Everything fired and survived together:

- **The #107 pick fired live**: ore jumps at 62 (row 115, v=80) and 61
  (row 118, v=101) — well below their window entries (~73/~92) — with
  downstream obstacles in next2; both chains survived.
- **Four steer slams** (ore@87, pit@66, pit@62, pit@86), all with
  surviving follow-ups — including the rebound-arc ore steer's first
  live save (the run-24 death class, now handled).
- **Four scoring slams** (PTS 1-4) + rescue jumps at dist 10 surviving
  three times; the gap-aware ceiling fired at 55 (the clamp) at v=129.

The death after ~18s: a forced-rescue cascade ending at a 72px-wide pit
— JUMP #22's rescue clicked on an already-decided landing. Scoring
stalled at 4 through the last ~6s (slam #13 fired but PTS held — worth a
look when refining). Remaining #107 items unchanged: touchdown
instrumentation for T_REB, conversion-slam gating.

### Next session

1–2. **DONE** (Runs 5–6): airborne detection; ore/obstacle classification.
3. **Spaced-obstacle / multi-jump policy** — the airborne guard stops the
   *lethal slam*, but the bot still needs the rapid *second jump* after
   landing. That needs a live `--save-frames` run capturing a multi-jump
   sequence to measure the inter-obstacle spacing and the land→re-arm timing
   (cooldown shortening + landing detection via `cart_y` returning to the
   grounded baseline, which `policy.GroundedBaseline` already tracks).
4. **Slam-on-ore scoring (Phase D)** and the **digit bootstrap (#6)** — both
   need a *scoring* `--watch` run (a human scores where the bot can't). That
   run also provides the first real slam-ore + a real static-vs-scrolling
   capture to (re)build the velocity filter and finalize the OCR box.

Trigger retuned to `[20,30]` (Run 11 analysis above): the old `[30,50]` fired
too early and landed into the pit's far edge. The remaining unknown is the gap
width W + the collision model, which the next `--save-frames` re-run resolves.

## Open questions

- ~~Does a slam onto an ore reliably rebound enough to clear the *next*
  pit, or is a fresh jump still needed?~~ **Answered (Run 15):** the
  rebound is a full jump-height arc (~82px apex) — no fresh jump needed;
  the human chained two slams off rebounds with zero grounded clicks.
- Ore colour → point value: worth prioritising high-value ore, or just
  slam every ore? (Decide after seeing point density.)
- Two-click jump→slam combo latency: is `random_delay`'s 80-200ms between
  the two clicks tight enough at high speed, or does the slam need a
  shorter, fixed inter-click gap?
