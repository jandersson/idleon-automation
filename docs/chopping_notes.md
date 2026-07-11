# Choppin minigame — mechanics research (2026-06-11)

Community-sourced mechanics model for the choppin minigame, gathered
because attempts are capped (5/day) and each one needs to count.
Cross-referenced against the bot's own first instrumented session
(2026-06-11 00:13, see TODO.md). Sources at the bottom; per-claim
confidence noted since none of this is from the game's code.

## Scoring

- **Green chop = +1 point, yellow/gold chop = +2.** (wiki + DigitalTQ,
  consistent everywhere; high confidence)
- Score thresholds that matter beyond per-attempt rewards: a quest
  wants 150, an achievement at 141 pays 25 gems + 4hr time candy, and
  **mage skill progression scales with the account's best score** —
  so peak score matters, not just average rewards. (Steam threads +
  DigitalTQ; medium-high confidence)
- Rewards per attempt: Choppin EXP, logs of that tree, leaves —
  scaling with points. **Needs ≥5% choppin efficiency on the
  character or the attempt pays nothing.** (wiki summary; medium
  confidence on the 5% number)

## Speed model (settled 2026-06-11, 01:08 session — 97s, 19 chops)

- **Chops ramp the speed; bounces do NOT — confirmed unconfounded by
  the pause experiment (10:51 session, 2026-06-11).** The bot paused
  firing 12s after the 5th and 10th registered chops; per-sweep peaks
  INSIDE the chop-free pauses were flat at both low and mid speed
  (pause 1: first-half mean ~455 → second-half ~433 px/s over ~20
  bounces; pause 2: ~576 → ~560 over ~25). This closes the original
  confound (the opening ramp had chops+bounces together; the 01:08
  droughts were at high speed where saturation could mask). The
  experiment stays available via BOUNCE_EXPERIMENT_EVERY_N (set 0 to
  disable now that it's answered). Corollary intact: waiting is free,
  deaths are the only cost.
- **Therefore waiting is FREE.** A skipped fire window costs
  wall-clock, not points or difficulty. Deaths are the only real cost
  in this game. The front-loading doctrine ("time not scoring is
  actively harmful") is WRONG and is hereby retracted.
- **No inactivity timeout** at least up to 45s (the longest drought —
  the round did not end). The 97s round ended only by the bot's own
  marginal fire.
- **Leaf motion is EASED**: mid-bar median ~550–750 px/s vs ~280–420
  near the edges (profile is trapezoid-ish — broad fast middle, slow
  edges). Gate consequence: instantaneous vx near the edges flatters
  time-to-red; the gate floors projected speed at
  EASING_SPEED_FLOOR_FRAC of the recent max (0.75 since the 01:08
  death — chop 20 fired at nominal ttr=150ms on vx=425 while the true
  crossing speed was ~630-750, real ttr ~100ms).
- **True-speed kill boundary ≈ 100–115ms**: all three deaths to date
  had true ttr ≤ ~110ms; survivals observed at 115–123ms.
- **Yellow chops slow the leaf down** ("Hitting yellow slows down the
  speed the mark moves") — community claim, still unconfirmed: 7 gold
  chops in the 01:08 session produced no visible dip (any brake may be
  cancelled by the chop's own ramp).

## Chop registration (measured, 00:46 session)

- A registered chop re-rolls the zone layout within **86–201ms** —
  the re-roll is the in-game ack.
- The re-roll is NOT the end of the game's chop cooldown: clicks
  198/201/225ms after a registered chop were **silently ignored** (no
  re-roll, no point), while 655/720/812ms gaps registered. The
  registration boundary sits somewhere in (225, 655)ms.
  UPDATE 2026-07-06 (runs 3-5, live PTS ground truth): **every
  click in those runs scored** (maintainer) — gaps down to 588ms all
  paid, so the boundary is < 588ms and the (225, 588) range stays
  unbisected (a same-evening 0.66 floor raise blaming 609-631ms gaps
  was wrong and was reverted). The recurring -3-per-run gap vs the
  +1/+2 model is instead **gold fires paying +1**: the PTS counter
  updates instantly, so run 5's truthful 23-after-chop-18 (vs 27
  modeled) plus final 31 (vs 34) mean ~3-4 of the gold-labeled fires
  were evaluated off-gold — 7 of its 8 ride-triggered gold fires
  were LEFTWARD entry fires (fire the instant the detected left edge
  crosses in), so the working hypothesis is the game's hit point
  sits a few px right of the sprite mask's leftmost pixel. Issue
  #110: measure the offset from `polls.pts_read` steps × fire depth
  (pointer_x + zone_layout, already logged), then add a depth-aware
  gold fire rule.
- **Ignored clicks are not free**: the 00:46 round's fatal click came
  225ms after its predecessor — no point was possible, but the death
  still happened. Working model: every click is evaluated against the
  logic-leaf position (green during cooldown → nothing; red → death,
  always). So clicks during the cooldown window carry pure downside —
  the fire hold now enforces the full interval.
- The fatal click itself: the leaf read the same x twice 127ms apart
  (stale/frozen render or capture) → vx = exactly 0.0 → the old
  `abs(vx) > 1e-6` check disarmed the directional gate → click landed
  with the real leaf likely ~355 px/s and red ~31px (~87ms) ahead.
  Direction-unknown fires are now banned (MIN_VX_FOR_FIRE).

## Round end

- **Clicking while the leaf is over red ends the round.** One
  mistake, no lives. (Steam "broken" thread + our chop #3 death:
  fired 19px ahead of red at ~50–75ms time-to-red, leaf vanished
  250ms later, bar gone.) No timer or chop-cap mentioned anywhere —
  rounds appear to end only by red-click. UNVERIFIED: whether
  inactivity ends a round, and whether a voluntary exit keeps the
  score.

## Zone dynamics

- **A successful chop shifts/re-rolls the zone layout under the
  leaf** ("double clicking moves it twice and you run the risk of
  clicking the red space that moves under the cursor" — Steam tips).
  Matches the bot's RLE observation: red grew r50→r53 / r45→r48 and
  green shrank 120→114 across three chops. Implication: any cached
  layout is stale the instant a chop registers — always re-sample
  before the next click (the loop already does).
- Gold zones exist in some layouts (seen once in a transition frame);
  unknown what makes them appear.

## Strategy (community consensus + implications for the bot)

1. **Front-load chops.** "Click multiple times when the arrow passes
   over the green at the start" is the community max-score line: the
   first seconds are the slowest and safest, and points-per-pass is
   highest before the speed ramp. IMPLEMENTED 2026-06-11: the fixed
   150ms post-chop sleep became a layout-settle hold — the bot
   re-arms the moment the zone layout re-rolls (the in-game signal
   the chop registered), with 150ms as the fallback deadline only.
   The polls record the actual settle lag (fired=1 row → first
   changed zone_layout), which calibrates whether the fallback can
   shrink further.
2. **Always take yellow.** +2 and a slowdown. IMPLEMENTED 2026-06-11
   as the same-sweep gold upgrade: a safe green fire is deferred when
   gold lies ahead of the leaf before any red (same sweep, no extra
   bounce, capped at GOLD_RIDE_MAX_MS); a leaf already over gold
   fires as before. Cross-pass gold waiting was deliberately NOT
   built — its value depends on the unresolved speed-attribution
   question below. UPDATE 2026-07-06: speed attribution settled
   per-chop, so cross-pass waiting is now IMPLEMENTED
   (`gold_wait.py`): with chops (not time) bounding the score, a
   green→gold upgrade is a pure +1 point, and the 2026-06-11 data
   showed 20/48 registered green chops fired with gold sitting
   elsewhere on the bar (gold is on the bar in ~95% of long-session
   polls, the leaf over it only ~10-15%). LESSON from the first live
   run (18:32 session, stalled at chop 9): the layout only re-rolls
   on a chop, so a red-flanked gold (that run: ~29ms from red, gate
   needs 120) stays unfireable until the next chop — and the
   same-sweep ride, which had NO deadline, held every safe green for
   ~20s until the failsafe. Fix: ride + wait share one chase clock
   (`update_gold_chase`); at MAX_CHASE_S=6s (~4 sweeps) the bot gives
   up on that layout's gold and takes greens until the next re-roll.
   Cross-pass waits also only START within START_LATEST_S=10s of the
   last fire, so the chase can't trip the 60s starve exit.
   Instrumented via `chops.gold_wait_ms` (chase duration preceding a
   fire; zone says whether it paid off) and `polls.hold_reason`
   ('gold_wait'/'gold_ride').
3. **Never click red** — the bot's one death condition. The
   directional time-to-red gate (MIN_TIME_TO_RED_MS) is the
   load-bearing control; late-round it will (correctly) starve fires
   as speed rises, at which point the attempt has reached its
   natural ceiling.
   UPDATE 2026-07-06: the gate's starve IS the score ceiling — at the
   ~600-650 px/s saturation a 74px green crosses in ~116ms, so no
   moment ever satisfies a 120ms-runway gate (runs 2-5 all ended
   there, 18-24 chops). Replaced as default by **planned shots**
   (`planner.py`): pick the time-domain center of an upcoming zone
   crossing, schedule the click for impact minus CLICK_LATENCY_S
   (40ms), re-plan every poll, spin-wait to the instant, fire.
   Impacts must clear a required number of sigmas from every red
   boundary and 1σ inside their own zone. Replay over run 5's polls:
   feasible plans on 37% of the starve-tail polls where the gate
   found nothing (median margin 62ms). Also fires from any leaf
   position and lands mid-zone (absorbs the #110 hit-point offset).
   The gate survives as CHOPPING_AIM=gate.
   CROSS-SWEEP (2026-07-07): the planner also targets zone crossings
   on the pass AFTER the upcoming bounce (every zone yields a sweep-0
   and a sweep-1 candidate; turnaround red pockets are crossed twice
   and both crossing times bound the margins). Error model, all
   measured: base σ 12ms (run-6 mid-bar fire residuals) × 1/sin(θ)
   at the target (edge fires ran +40ms) + 5% of the horizon (V_max
   drift); required sigmas ladder 3.0/2.5/2.25 by drought time (EV:
   skipping a 3σ gold costs more expected points than the ~0.1%
   death it avoids — and mid-drought, deaths bank anyway). Impact
   placement scans fractions of the crossing, not just the center,
   so a one-sided red buys margin by shifting away. chops.plan_sweep
   counts how often through-bounce fires happen. CLICK_LATENCY_S is
   55ms since run 6's post-mortem.
4. Edge bounces can't be prevented (leaf motion is autonomous), so
   "avoid the sides" translates for a bot into: score as much as
   possible per unit time early, and bank yellow slowdowns.

## Open unknowns

- ~~Chop-vs-bounce speed attribution~~ — settled: per-chop (above).
- ~~Does inactivity end a round?~~ — no, at least ≤45s.
- ~~Does a voluntary exit keep the score?~~ — YES (maintainer,
  2026-06-11): points bank as in-game tokens on exit; each game
  starts fresh. Exit-on-starve implemented (STARVE_EXIT_S=60): after
  60s without a safe fire the bot ends the session with the score
  summary and the user exits in-game to bank. Every bot death so far
  came from finally taking a marginal window during a starve — this
  removes that failure mode entirely.
- Minimum inter-chop delay: all 560ms+ gaps registered; the (225,
  560)ms range is untested but moot while the fire cadence is
  crossing-limited anyway.
- What spawns gold zones (first gold appeared at chop 7-8 in the
  01:08 session and persisted; gold width shrank o41 → o33 over the
  round).
- Whether the per-chop ramp keeps compounding at higher scores
  ("hyperspeed" reports) or truly saturates.
- ~~Position-aware eased time-to-red~~ — BUILT 2026-06-11, after the
  01:39 starve-at-23-points proved the flat 0.75×recent-max floor was
  a bug, not the game's ceiling (maintainer's human best is 66 by
  just clicking greens). Root cause: detection-jitter spikes of
  1600–1800 px/s (physically impossible 45px inter-poll jumps)
  poisoned the recent-MAX for 3s at a time, pricing windows at ~190px
  of needed runway when greens are 70–110px. The eased model
  (detector.eased_time_to_red_ms, V_max = median of sin-corrected
  samples via detector.infer_vmax) is jitter-immune and
  position-aware. Back-test over all 45 recorded fires
  (scripts/validate_chop_gate.py): both computable deaths price at
  28ms and 110ms (skip), 40/41 survivals price ≥116ms (fire) — the
  kill boundary is bracketed at 110–116ms and the budget sits at
  120ms. The late-round fires the old floor starved out price at
  135–159ms and now fire.
- **PTS ground truth (2026-06-11):** the overlay renders a live
  "N PTS" counter (left of the bar) and the account best ("66 BEST",
  right). The bot OCRs PTS once per fire hold (~0.35s post-click,
  zero throughput cost) into `chops.pts_after_ocr` — per-chop
  increments verify the +1/+2 mapping — and reports the in-game
  score at session end. Requires a one-time
  `chopping-pick-score-region` around the PTS digits.

## Sources

- IdleOn wiki — Choppin: <https://idleon.wiki/wiki/Choppin> (page is
  Cloudflare-gated to scripts; content via search summaries)
- DigitalTQ Choppin guide: <https://www.digitaltq.com/wiki/idleon/choppin-guide>
- Steam: "Tips for minigames": <https://steamcommunity.com/app/1476970/discussions/0/3807284068743364838/>
- Steam: "The chopping and mining mini-game are not fun or completely
  broken": <https://steamcommunity.com/app/1476970/discussions/0/3167694551647775610/>
