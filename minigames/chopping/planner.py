"""Planned-shot aiming for the chopping bot (2026-07-06 refactor).

The reactive gate (fire only when the leaf is ALREADY inside green/gold
with MIN_TIME_TO_RED_MS of runway left) is the bot's score ceiling: at
the ~600-650 px/s speed saturation a 74px green crosses in ~116ms, so
no moment ever satisfies a 120ms-runway gate and the round starves
(runs 2-5 all ended that way). A human doesn't starve there — they time
the click for where the leaf WILL be.

This module does the same: given the zone layout and the leaf's state,
pick a target impact — the TIME-DOMAIN center of an upcoming zone
crossing (time-domain, not pixel: under eased motion the safe window is
asymmetric in pixels but symmetric in time, and timing error is what
kills) — and return when the leaf gets there. The caller schedules the
click for impact minus its click latency and re-plans every poll until
commitment.

Side benefits over the gate:
- fires can be initiated from ANY leaf position (over red/none too),
  not just already-in-zone polls;
- impacts land mid-zone, so the #110 hit-point offset (gold entry
  fires paying +1) is absorbed by tens of px of margin instead of
  needing to be measured first;
- the same margins absorb the ±3px click jitter and detection noise.

Safety is expressed as time margins: the impact must sit at least
red_sigmas x sigma_ms from every red boundary crossing (deaths are the
only real cost), and at least 1 sigma inside its own zone (missing
into an adjacent green/none just wastes or downgrades a chop — cheap).
Pure module: no IO, unit-testable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Zone:
    kind: str  # 'g' green, 'o' gold, 'r' red, 'n' none/uncolored
    x0: int    # inclusive left column
    x1: int    # exclusive right column

    @property
    def width(self) -> int:
        return self.x1 - self.x0


def parse_layout(layout: str) -> list[Zone]:
    """Parse detector.zone_layout's RLE string ("n4 g58 r12 o9 ...")
    into Zone spans, left to right."""
    zones: list[Zone] = []
    x = 0
    for run in layout.split():
        kind, width = run[0], int(run[1:])
        zones.append(Zone(kind, x, x + width))
        x += width
    return zones


@dataclass(frozen=True)
class Plan:
    target_x: float      # predicted leaf x at impact (bar coords)
    impact_in_s: float   # seconds from plan time until impact
    zone_kind: str       # 'g' or 'o' — what the impact scores
    value: int           # 1 (green) or 2 (gold)
    margin_ms: float     # min time-margin from impact to a red boundary
    sweep: int = 0       # 0 = current sweep, 1 = after the next bounce


# Empirical death-risk curve (2026-07-11, calibrated on the three runs
# using the current sigma model: 87 planned fires, 2 deaths — BOTH at
# floor margins, 3.05 and 2.54 sigmas; 2/30 below 3.5 sigmas = 6.7%,
# 0/57 above). The real tail is ~50x fatter than Gaussian at the floor
# (model breakdowns, not sampling noise), so risk is priced from this
# fit rather than normal tables: P(death | k sigmas) = exp(A - B*k),
# anchored at P(3)=0.08 and P(5)=0.005, capped at 0.5.
RISK_A = 1.63
RISK_B = 1.39


def death_risk(sigmas: float) -> float:
    """Empirical per-shot death probability at a margin of `sigmas`."""
    return min(0.5, math.exp(RISK_A - RISK_B * sigmas))


# Horizon-proportional error term (2026-07-07, cross-sweep planning):
# a fractional V_max mis-estimate shifts arrival by that fraction of
# the whole horizon, so distant impacts carry proportionally more
# timing risk. 0.05 ~= the p75 V_max estimator's observed jitter; at
# run 6's 40-120ms same-sweep horizons it adds 2-6ms (consistent with
# the measured 17.4ms total vs the 16ms base), while a 1s next-sweep
# horizon adds 50ms — distant cross-sweep plans self-reject until the
# per-poll replanning brings them close.
HORIZON_ERR_FRAC = 0.05


# Floor for the position factor in _sigma_scale: past sin(theta)=0.35
# (the outer ~3% of the bar) the model isn't trusted at any margin.
MIN_TARGET_SIN = 0.35


def _sigma_scale(target_x: float, bar_w: int) -> float:
    """Position multiplier for the timing-error budget: 1/sin(theta) at
    the target. Near the turnarounds the leaf's local speed drops, so a
    fixed position error costs proportionally more TIME — measured on
    the first planner run (2026-07-06 23:44): arrival residuals were
    ±10-20ms for mid-bar targets but +38/+41ms for both sin(theta)≈0.6
    targets, and the one death was a sin(theta)=0.61 target whose flat
    margin had no room for that."""
    c = 1.0 - 2.0 * target_x / bar_w
    s = math.sqrt(max(0.0, 1.0 - c * c))
    return 1.0 / max(s, MIN_TARGET_SIN)


def plan_shot(
    zones: list[Zone],
    x: float,
    direction: float,
    v_max: float,
    bar_w: int,
    earliest_impact_s: float = 0.0,
    sigma_ms: float = 16.0,
    red_sigmas: float = 3.0,
    max_horizon_s: float = 2.5,
    horizon_err_frac: float = HORIZON_ERR_FRAC,
    death_cost_pts: float = 0.0,
) -> Plan | None:
    """Pick the best planned impact on the current sweep OR the next
    one (through the upcoming bounce), or None.

    Every green/gold zone yields up to two candidates: its remaining
    crossing this sweep (sweep=0, if any of it lies ahead) and its full
    re-cross after the turnaround (sweep=1). Each candidate impact is
    the time-domain center of its crossing, shifted later if the
    caller's earliest_impact_s (cooldown release + click latency)
    demands it. A candidate is feasible when:
    - its time-margin to every red boundary CROSSING (both sweeps —
      turnaround red pockets are crossed twice and both times count)
      is >= red_sigmas * sigma_local;
    - it sits >= 1 sigma_local inside its own zone (missing into a
      benign neighbour only wastes a chop);
    - it lands within max_horizon_s.

    sigma_local = sigma_ms * _sigma_scale(target) + horizon_err_frac *
    horizon: edge impacts and distant impacts both demand more margin
    (see the constants' comments).

    Ranking (2026-07-11, the floor-margin gold-vs-green pass): feasible
    candidates rank by EXPECTED VALUE — value minus death_cost_pts *
    death_risk(sigmas) — then by sigmas, then sooner. With the
    calibrated risk curve a floor-margin gold (+2 at ~8% death) loses
    to a fat-margin green (+1 at ~0) whenever the remaining round is
    worth more than ~12 points, which is exactly the trade that killed
    runs 12 and 13. death_cost_pts=0 (the default) reduces to the old
    value-first ranking.

    EV REORDERS but never refuses: a lone rung-passing candidate fires
    regardless of its EV. The layout is static between chops, so the
    alternative to a lone candidate isn't "a better shot later" — it's
    the starve exit, which banks the same points a death would (run 14's
    chop 9: a forced 3.1-sigma gold with no green anywhere was the
    round's only path forward; firing at ~7% death risk dominates
    starving at 100% round-end). The hard sigma rungs remain the only
    absolute refusal.
    """
    if v_max <= 0 or bar_w <= 0 or direction == 0:
        return None
    sigma_s = sigma_ms / 1000.0
    omega = 2.0 * v_max / bar_w

    def mirror(p: float, d: float) -> float:
        return p if d > 0 else bar_w - p

    def theta(pm: float) -> float:
        return math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * pm / bar_w)))

    theta0 = theta(mirror(x, direction))
    t_turn = (math.pi - theta0) / omega

    def t_this_sweep(p: float) -> float | None:
        """Time to reach p on the current sweep; None if p is behind."""
        tp = theta(mirror(p, direction))
        return (tp - theta0) / omega if tp >= theta0 - 1e-9 else None

    def t_next_sweep(p: float) -> float:
        """Time to reach p on the pass after the upcoming bounce."""
        return t_turn + theta(mirror(p, -direction)) / omega

    def position_at(t_s: float) -> float:
        """Leaf position t_s seconds out, through at most one bounce."""
        if t_s <= t_turn:
            th1 = theta0 + omega * t_s
            pm = bar_w / 2.0 * (1.0 - math.cos(min(math.pi, th1)))
            return pm if direction > 0 else bar_w - pm
        th2 = min(math.pi, omega * (t_s - t_turn))
        pm = bar_w / 2.0 * (1.0 - math.cos(th2))
        return pm if -direction > 0 else bar_w - pm

    # Every red boundary crossing TIME over both sweeps.
    red_times: list[float] = []
    for z in zones:
        if z.kind != "r":
            continue
        for e in (float(z.x0), float(z.x1)):
            t0 = t_this_sweep(e)
            if t0 is not None and t0 > 0:
                red_times.append(t0)
            red_times.append(t_next_sweep(e))

    best: Plan | None = None
    best_key: tuple = ()
    for z in zones:
        if z.kind not in ("g", "o"):
            continue
        candidates: list[tuple[float, float, int]] = []
        # Sweep 0: the remaining crossing ahead of the leaf.
        if direction > 0:
            a0, b0 = max(float(z.x0), x), float(z.x1)
        else:
            a0, b0 = min(float(z.x1), x), float(z.x0)
        if (b0 - a0) * direction > 0:
            ta, tb = t_this_sweep(a0), t_this_sweep(b0)
            if ta is not None and tb is not None and tb > ta:
                candidates.append((ta, tb, 0))
        # Sweep 1: the full re-cross after the bounce (entered from the
        # far side, so the edge order flips).
        entry, exit_ = (float(z.x1), float(z.x0)) if direction > 0 \
            else (float(z.x0), float(z.x1))
        ta1, tb1 = t_next_sweep(entry), t_next_sweep(exit_)
        if tb1 > ta1:
            candidates.append((ta1, tb1, 1))

        value = 2 if z.kind == "o" else 1
        for t_a, t_b, sweep in candidates:
            # Impact placement: the crossing center is the default, but
            # when red abuts one side only, a shifted impact buys real
            # margin — try a few fractions of the crossing and keep the
            # statistically safest feasible one.
            for frac in (0.5, 0.3, 0.7, 0.15, 0.85):
                t_star = max(t_a + frac * (t_b - t_a), earliest_impact_s)
                if t_star > max_horizon_s:
                    continue
                target_x = position_at(t_star)
                sigma_local_s = (sigma_s * _sigma_scale(target_x, bar_w)
                                 + horizon_err_frac * t_star)
                # Must stay >= 1 (local) sigma inside the zone.
                if t_star < t_a + sigma_local_s or t_star > t_b - sigma_local_s:
                    continue
                margin_s = min((abs(rt - t_star) for rt in red_times),
                               default=math.inf)
                sigmas = margin_s / sigma_local_s
                if sigmas < red_sigmas:
                    continue
                margin_ms = margin_s * 1000 if margin_s != math.inf else 10.0**6
                # Rank by expected value (see docstring), then safety,
                # then sooner.
                ev = value - death_cost_pts * death_risk(sigmas)
                key = (ev, sigmas, -t_star)
                if best is None or key > best_key:
                    best = Plan(target_x, t_star, z.kind, value, margin_ms, sweep)
                    best_key = key
    return best
