# Catching: the calibrated phase-timing model (#60)

How the catching bot decides *when* to flap, the dynamics fit from a dense
trajectory trace, and the open control problem. The loop + fallback live in
`main.py`; the model in `controller.py`; the offline fit in `fit_dynamics.py`;
a closed-loop check in `sim.py`.

## The control problem

Flappy Bird with hoops: a bobbing avatar at a fixed screen-x, golden hoops
scrolling in from the right, one click = one flap. "If the ground or the edge
of a ring is hit, that game is over" (wiki) — so the avatar must thread *every*
hoop; one clip ends the run. The hard constraint, measured:

- A flap's impulse gives a **~44px** bob (peak-to-peak per flap).
- A hoop's **passable hole is ~30px**.

So the bob is bigger than the hole's slack. The avatar can't hover inside the
hole; it must cross at a moment its excursion stays within the hole.

## Dynamics (fit + verified against the 2026-06-17 dense trace)

Screen y grows downward. Between flaps the avatar free-falls at constant
gravity `g`. A flap is a **set-velocity impulse**: it sets vy to `flap_vy`
(< 0 = up), independent of the incoming velocity. Hoops scroll left at constant
`approach_speed`. Fit from one dense `--trace` run and adversarially verified
(5 independent re-estimators):

| param | value | how |
|---|---|---|
| gravity | **388 px/s²** | velocity-slope over free-fall arcs (= whole-run position-quadratic; both agree) |
| flap_vy | **−185 px/s** | apex-rise: `-√(2·g·rise)`, rise ≈ 44px |
| approach_speed | **78 px/s** | slope of `gap_left_x` over long monotonic runs |

Self-consistent: `rise = flap_vy²/(2g) ≈ 44px`, `time_to_apex = |flap_vy|/g ≈
0.48s`, full bob ≈ 0.95s.

**The bob is a SYMMETRIC ballistic arc** (~0.48s up, ~0.48s down). Earlier notes
claimed "fast rise ~0.16s, slow descent ~0.9s" — that was an artifact of the
double-tap hover flaps, *not* a single arc. The 0.9s was the full bob.

### Fitting (`catching-fit-dynamics`)

Reads the newest `assets/traces/trace_*.csv` and writes `assets/dynamics.json`.
- **gravity** — median of per-arc fits over free-fall runs.
- **flap_vy** — apex-rise (`-√(2·g·rise)`). NOT the first post-flap finite
  difference: at 37Hz / integer pixels that's a poll-AVERAGED velocity biased
  toward zero by ~g·dt/2 (it read −131 for a true −185).
- **approach_speed** — `−slope` of `gap_left_x` over the long monotonic runs
  (`gap_left_x` saturates/clamps at fixed values when far; those plateaus are
  excluded by the slope<0 filter).

## Firing rule — apex-threading

The bob (~88px peak-to-peak of travel, ~44px each way) is bigger than the
passable hole, so the avatar can only stay inside the hole through a crossing
if it's near its **apex** (vy≈0, slowest) as the hoop passes — at mid-bob speed
(~185px/s) it moves ~65px during the 0.31s crossing, far outside a 30px hole;
near the apex it drifts only ~6px. So `should_flap_now` fires so the apex lands
at the crossing midpoint: when the hoop is one apex-time away
(`t_enter ≤ time_to_apex ≤ t_exit`) AND this flap's apex (`fly_y − rise`) lands
inside the passable opening (the detected band inset by the rim thickness,
`RING_INSET_PX`, then by the crossing excursion). Crossing **mid-descent** —
the original `should_flap_now` — is wrong: the avatar zips through the hole and
clips. `descent_time_to`/`predict_descent_window` remain only for analysis.

## The open problem: phase-locking (why threading isn't solved yet)

`sim.py` runs this exact policy against the fitted dynamics and threads **0**:
the launch only fires when the avatar *happens* to be at launch height during
the timing window, which rarely coincides. The root cause is structural:

> With a fixed-impulse flap, the bob **period is constant** (`P = 2·t_apex`,
> ~0.95s) regardless of flap height. So you cannot independently set the apex
> **height** (which needs launching from `launch_y = hole_center + rise`) and
> the apex **time** (which needs the launch at `t_cross_mid − t_apex`).

Aligning both requires **active multi-bob phase-locking**: absorb the phase
residual `(t_cross_mid − t_apex − natural_bob_bottom) mod P` over several
off-level bobs (off-center apexes between hoops are harmless), landing a
`launch_y` bob-bottom exactly on `t_cross_mid − t_apex`. The apex region is
forgiving (the avatar stays within ±15px of the apex for ~±0.28s, so the
crossing must align within ~±0.12s), so the lock needn't be precise — but it
must exist. Naive variants (passive gate, receding-horizon hold) either never
fire or overshoot the ceiling/clip the top; see the sim.

**This is best tuned with live data**, not blind sim-fitting: the real
click→game-response latency and any dynamics drift shift the timing by tens of
ms, which is inside the tolerance budget.

### Run 16 (first live `--model`, 2026-06-17): 23.8s, scored 2 — longest yet

What worked: the startup fix held (no early sink), and the avatar's apexes
clustered at **y≈86–91, right on the hole centre (gap midpoint ≈89)** — so the
apex-hover puts the avatar at the correct *height*. Survival is real.

Two bugs the trace exposed (both now fixed):
- **Branch order starved the launch.** `floor_rescue` was checked before the
  model launch, and both fire in the same low-in-the-bob zone — so floor won
  22:2 over the model, and a floor flap sets no coast, so the avatar over-lifts
  into the next crossing and clips the **top** (the death: an apex at y=44 vs a
  hole top of 52). Fix: check the launch first (it also flaps up, so it's
  floor-safe); floor backstops only the polls with no launch due.
- **Floor lookahead too long.** The 0.1s predictive horizon fired floor ~18px
  early (at y≈112, before the avatar reached the launch zone at 117+),
  preempting the model. Cut to 0.05s (~1–2 polls at 37Hz); detection-gap falls
  are caught by the position bound, not the lookahead.

### Run 17 (branch-order fix live, 2026-06-17): 52.5s, scored 12

Big jump (2 → 12) — the reorder let the model launch fire and the floor rescue
threaded the orange rings (its fixed bound ≈ 130 anchors the apex at ~79, near
the orange centre ~87). It died at a **green ring**: green rings sit *higher*
(centre ~73 vs orange ~87) and are smaller, and the `FLAP_MARGIN` on the model
hover delayed the hover flap ~6px so the floor rescue preempted it and anchored
the apex low (apex 88 vs the green centre 73 → bottom clip). Fix: the model
hover flaps AT the bob bottom (no margin), so the hover beats the floor on the
higher rings and the apex centres (probe: apex offset at centre 73 goes +6 →
+2; at 65, +9 → +2). Orange is unaffected (floor still anchors it). Known
follow-up (#52): `classify_hoop_color` counts the whole bbox, so a green ring
over the orange cliff misclassifies as orange — it needs the contour mask, not
the bbox (it's not yet wired into gameplay, so it doesn't affect threading).

### Why the phase-lock is hard (sub-bob granularity)

You can't hover tighter than the 44px bob: every flap imparts a fixed ~44px of
rise, so the avatar always traverses ≥44px per cycle — apex-threading is
mandatory, confirmed. To land an apex on a crossing you must shift the bob
phase, but **one flap = one full period P (~0.95s)**, so a sub-P phase residual
can't be absorbed by a single flap. Options (untried, for the next pass): a
short off-period bob (let the avatar fall δ below `launch_y`, flapping late →
apex slightly low but in-hole, delaying the bottom by a bounded amount), spread
over 1–2 bobs; or a search-based planner over discrete flap decisions. Both are
best validated against the next live trace (does the launch land the apex on
the crossing under real latency?), not the sim alone.

Next: run `catching --model --trace` again with the branch-order fix; from the
trace, check whether the now-more-frequent model launches land apexes on
crossings, and whether it threads past 2.

## Floor safety (shipped, helps both paths)

The 2026-06-17 trace run died by **sinking to the floor**: the avatar free-fell
~100px over 1.1s and a position-threshold hover flap fired too late (the avatar
crossed the threshold at ~450px/s and overshot to the floor in one poll).
`floor_rescue_due` is a predictive backstop — flap if `fly_y` is below
`frac·play_height` OR a fast descent is projected to cross it within a short
lookahead. It runs FIRST (before coast/hover/model), so a held coast or a
detection gap can't let the avatar sink. Pure kinematics, so it guards the
hand-tuned default path too.

## What the model can and can't do

The bob/hole ratio and the in-game **speed ramp** cap every run regardless. The
corrected dynamics + apex-threading + floor safety remove the knife-edge and
the floor sink; reliable threading of the early hoops waits on the phase-lock.
Expect: survives the floor sink; threads opportunistically until the phase-lock
lands. Don't expect an unbounded score.

## Iterating

1. `catching --trace --save-frames` → `catching-fit-dynamics` (writes/refreshes
   `dynamics.json`); eyeball the fit against the trace.
2. `catching-sim` to check a controller change threads in-model before a live
   run (compare variants; absolute counts are indicative — the geometry consts
   are estimates).
3. `catching --model --trace` live; inspect the `where='model'` flaps and the
   dense trace to see whether apexes land on crossings. Tune the phase-lock to
   the live latency. Iterate on the trace, not single-pixel constant nudges.
