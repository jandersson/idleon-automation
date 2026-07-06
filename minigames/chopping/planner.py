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

from minigames.chopping.detector import eased_time_to_x_s


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


def _sweep_times(
    x: float, direction: float, positions: list[float],
    v_max: float, bar_w: int,
) -> list[float] | None:
    """Seconds until the leaf reaches each position (all ahead of x in
    `direction`), under the eased model. None when unusable."""
    out = []
    for p in positions:
        d = (p - x) if direction > 0 else (x - p)
        t = eased_time_to_x_s(x, direction, d, v_max, bar_w)
        if t is None:
            return None
        out.append(t)
    return out


def plan_shot(
    zones: list[Zone],
    x: float,
    direction: float,
    v_max: float,
    bar_w: int,
    earliest_impact_s: float = 0.0,
    sigma_ms: float = 16.0,
    red_sigmas: float = 3.0,
    max_horizon_s: float = 1.2,
) -> Plan | None:
    """Pick the best planned impact on the CURRENT sweep, or None.

    For every green/gold zone ahead of the leaf, the candidate impact
    is the time-domain center of the zone's crossing, shifted later if
    the caller's earliest_impact_s (chop-cooldown release + click
    latency) demands it. A candidate is feasible when:
    - its time-margin to every red boundary is >= red_sigmas * sigma_ms
      (both the red behind the impact and the red ahead — deaths are
      the only real cost);
    - it sits >= 1 sigma inside its own zone (missing into a benign
      neighbour only wastes a chop);
    - it lands within max_horizon_s (velocity estimates decay; the
      caller re-plans every poll anyway).

    Feasible candidates rank by (value, margin): gold beats green, and
    among equals the safer impact wins.
    """
    if v_max <= 0 or bar_w <= 0 or direction == 0:
        return None
    sigma_s = sigma_ms / 1000.0
    red_margin_s = red_sigmas * sigma_s

    # Red boundary positions ahead of x (zone edges where red starts or
    # ends), for margin computation.
    red_edges: list[float] = []
    for z in zones:
        if z.kind == "r":
            red_edges.extend([float(z.x0), float(z.x1)])

    best: Plan | None = None
    for z in zones:
        if z.kind not in ("g", "o"):
            continue
        # Usable span of this zone ahead of the leaf, this sweep.
        if direction > 0:
            a, b = max(float(z.x0), x), float(z.x1)
        else:
            a, b = min(float(z.x1), x), float(z.x0)
        if (b - a) * direction <= 0:
            continue  # entirely behind the leaf
        times = _sweep_times(x, direction, [a, b], v_max, bar_w)
        if times is None:
            continue
        t_a, t_b = times
        if t_b <= t_a:
            continue  # degenerate (e.g. past the turnaround clamp)
        # Time-domain center, shifted for the cooldown if needed.
        t_star = max((t_a + t_b) / 2.0, earliest_impact_s)
        # Must stay >= 1 sigma inside the zone.
        if t_star < t_a + sigma_s or t_star > t_b - sigma_s:
            continue
        if t_star > max_horizon_s:
            continue
        # Predicted impact position: invert the eased model by bisecting
        # position along the span (monotone in time).
        target_x = _position_at(x, direction, t_star, v_max, bar_w)
        if target_x is None:
            continue
        # Margin to red: nearest red boundary crossing in time.
        margin_s = math.inf
        red_times = _sweep_times(
            x, direction,
            [e for e in red_edges if (e - x) * direction > 0],
            v_max, bar_w,
        )
        if red_times:
            margin_s = min(abs(rt - t_star) for rt in red_times)
        if margin_s < red_margin_s:
            continue
        margin_ms = margin_s * 1000 if margin_s != math.inf else 10.0**6
        value = 2 if z.kind == "o" else 1
        cand = Plan(target_x, t_star, z.kind, value, margin_ms)
        if best is None or (cand.value, cand.margin_ms) > (best.value, best.margin_ms):
            best = cand
    return best


def _position_at(
    x: float, direction: float, t_s: float, v_max: float, bar_w: int,
) -> float | None:
    """Leaf position t_s seconds from now under the eased model
    (same sweep; clamps at the turnaround)."""
    if v_max <= 0 or bar_w <= 0:
        return None
    xn = x if direction > 0 else bar_w - x
    xn = max(0.0, min(float(bar_w), xn))
    omega = 2.0 * v_max / bar_w
    theta0 = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * xn / bar_w)))
    theta1 = min(math.pi, theta0 + omega * t_s)
    pos = bar_w / 2.0 * (1.0 - math.cos(theta1))
    return pos if direction > 0 else bar_w - pos
