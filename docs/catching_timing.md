# Catching: the calibrated phase-timing model (#60)

How the predictive flap timer (`minigames/catching/controller.py`) decides
*when* to fire the launch flap, why it's structured that way, and how to fit
and iterate it. The bot loop and the hover/coast fallback live in `main.py`;
this doc is the model behind `catching --model`.

## The control problem

The catching minigame is Flappy Bird with hoops: a bobbing avatar at a fixed
screen-x, golden hoops scrolling in from the right, one click = one flap. The
hard constraint, measured from saved frames:

- A flap's impulse gives a **~55px peak-to-peak bob**.
- A hoop's **passable hole is ~30px**.

So the avatar is bigger than the hole's slack — it cannot *hover* inside a
ring. It has to **coast through**: time a flap so the avatar's slow descent
lines up with the moment the hoop crosses its x. The bob is asymmetric — a
fast rise (~0.16s) then a long descent (~0.9s) — and that long descent is the
threading window.

The old controller hand-picked three constants (`LEAD_DIST_PX`, `COAST_S`,
`TIMED_LOW_OFFSET_PX`) and sat on a knife's edge: a ~9px change in the avatar's
y when the timed flap fires flipped thread-vs-crash. This model replaces the
guesswork with dynamics fit from a real trajectory.

## Physics model

Screen y grows downward. Between flaps the avatar is in free fall under a
constant gravity `g` (px/s²). A flap is a **set-velocity impulse**: it sets the
vertical velocity to `flap_vy` (negative = upward), independent of the incoming
velocity — standard Flappy physics. Hoops scroll left at a constant
`approach_speed` (px/s). Three parameters: `(g, flap_vy, approach_speed)`.

Position after a flap from `y0` (with `vy0 = flap_vy`):

```
y(t)  = y0 + vy0·t + ½·g·t²
vy(t) = vy0 + g·t
```

- Apex (vy=0) at `t_apex = |flap_vy| / g`, height `flap_vy² / (2g)` above `y0`.
- Descent time to a target `Y` (the later, descending root):
  `t = (−flap_vy + √(flap_vy² − 2g(y0 − Y))) / g`,
  real only when `Y ≥ apex` (the avatar actually descends that far).

## Firing rule

Let **T** = the time for a *flap-now* to descend to the hole centre
(`descent_time_to`). T depends on the avatar's current y and the dynamics, and
is roughly constant poll-to-poll. Let **t_mid** = the crossing midpoint — when
the hoop's horizontal span `[gap_left_x, gap_right_x]` is centred on the
avatar's fixed x. As the hoop approaches, t_mid shrinks while T stays put.

**Fire the launch flap at the crossover `t_mid ≤ T`.** At that instant a
flap-now lands the descent-centre arrival right as the hoop crosses; because
the descent is slow, the avatar lingers in the hole across the brief crossing.
Firing earlier would drop the avatar through the hole before the hoop arrives;
later, after it's gone.

`should_flap_now` returns False (deferring to `main.py`'s hover/coast) when
there's no hoop, the approach speed isn't fit, the hoop has already passed, or
a flap-now can't reach the hole on descent (`T` undefined — too weak a flap or
the avatar too low). `main.py` keeps the coast-rescue and hover so the model
can't regress below the working ~2-hoop behaviour.

## Fitting from a `--trace` CSV

`catching --trace` writes one row per poll (t, fly_y, fly_vy, gap extent,
fired, where) to `assets/traces/` (gitignored). `catching-fit-dynamics` reads
the newest trace and recovers the three parameters (`controller.fit_dynamics`),
writing `assets/dynamics.json`:

- **gravity** — median over free-fall runs (consecutive non-flap polls),
  least-squares fitting `y(t) = c₀ + c₁t + ½g·t²` per run.
- **flap_vy** — median of the most-negative `fly_vy` in the 0.25s after each
  fired poll (the launch impulse). The observed value is the finite-difference
  average over the flap step, ~`flap_vy + ½g·dt`, a few px shy of the true set
  velocity — fine for timing.
- **approach_speed** — median of `−slope` over within-hoop runs where
  `gap_left_x` decreases monotonically (runs break when the detector drops the
  hoop or the "next hoop" switches and `gap_left_x` jumps right).

`fit_dynamics` returns None (and the bot keeps the hand-tuned fallback) if any
parameter can't be recovered from the trace — better no model than a bad one.

## What the model can and can't do

The bob/hole ratio and the in-game **speed ramp** still cap every run — hoops
arrive faster over time until one crosses inside a single bob period and can't
be threaded. The model makes the **early** hoops reliable by removing the
hand-tuned timing's knife-edge; it does **not** enable an unbounded score.
Expect more consistent threading of the first hoops, not a different ceiling.

## Iterating

Validation is live — there's no game simulator. The loop:

1. `catching --trace --save-frames` for one run.
2. `catching-fit-dynamics` → inspect the printed `(g, flap_vy, approach_speed)`
   and rise/apex against the frames; the trace CSV is the ground truth.
3. `catching --model` and watch threading; cross-check the `where='model'`
   flaps in `catching.db` and the dense trace.
4. If a fit parameter looks off, the trace shows which segment was thin —
   adjust the segment thresholds in `controller.py` or capture a longer trace.
   Iterate on the trace, not single-pixel constant nudges.
