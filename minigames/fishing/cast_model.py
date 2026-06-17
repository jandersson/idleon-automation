"""Fishing-minigame planning logic — pure, no Tk / no IO.

The fishing minigame is a HOLD-to-cast distance game (wiki: Fishing
Minigame): hold the mouse to charge, release to cast the lure a distance
set by the hold duration, landing it on a fish (Green 1 / Eel 2 / Squid 3
/ Whale 5 points) while avoiding mines. So the bot's control variable is
the hold time, and its core learned model is hold_ms <-> cast distance.

This module holds the two pure pieces:

1. `CastModel` + `fit_cast_model` — a linear hold_ms -> landing-distance
   fit over logged (hold_ms, landed_dist_px) casts, plus the inverse
   `hold_for_distance` the bot uses to aim. Linear is the simplest
   defensible starting form; if observe data shows curvature (a charge
   cap, easing), swap in a monotone spline here without touching callers.
   Mirrors the darts stripe_model precedent: fitted once at startup from
   the DB, exploration casts feed the next fit.

2. `choose_target` — pick which fish to cast at: highest point value that
   the cast model can actually reach, nearest as the tie-break.

CALIBRATION: `fit_cast_model` returns None until MIN_SAMPLES clean casts
exist, so the bot explores (random holds, logged) until the relationship
is pinned. See docs/fishing_minigame.md.
"""
from __future__ import annotations

from dataclasses import dataclass

# Point value per fish kind (wiki Fishing Minigame). 0 = miss / mine-only.
FISH_VALUE: dict[str, int] = {
    "green": 1,   # Green Fish
    "eel": 2,     # Eel (after a 3-green streak)
    "squid": 3,   # Squid
    "whale": 5,   # Whale (catching it resets the streak to 1)
    "megalodon": 0,  # behemoth trophy fish; not a scoring target by default
}

# Casts needed before the linear fit is trusted over exploration. The
# relationship is 1-D and smooth, so this is modest; raise if observe data
# is noisy.
MIN_SAMPLES = 12


@dataclass
class CastModel:
    """Linear hold_ms -> landing-distance fit (distance in px from the
    cast origin). `hold_min_ms`/`hold_max_ms` bound the observed support;
    the bot never holds outside it (the fit can't be trusted where no cast
    has been sampled — the darts pose-support lesson)."""
    slope: float       # px gained per ms of hold
    intercept: float   # px at hold_ms = 0 (lure's minimum throw)
    hold_min_ms: int
    hold_max_ms: int
    n: int

    def reach_px(self) -> tuple[float, float]:
        """(min, max) landing distance the model can aim for, i.e. the
        distances at the support's hold bounds."""
        return (
            self.slope * self.hold_min_ms + self.intercept,
            self.slope * self.hold_max_ms + self.intercept,
        )


def fit_cast_model(
    samples: list[tuple[int, float]],
    min_samples: int = MIN_SAMPLES,
) -> CastModel | None:
    """Robust (Theil-Sen) line through (hold_ms, landed_dist_px) samples.

    The landing measurement is noisy — the bobber poll occasionally catches a
    launch/mid-arc position instead of the landing, producing outliers (a long
    hold with a tiny distance). Least-squares would let those skew the aim, so
    the slope is the MEDIAN of all pairwise slopes (Theil-Sen, robust to ~29%
    outliers) and the intercept the median residual.

    Returns None below `min_samples`, when the holds don't vary (degenerate),
    or when the robust slope is non-positive (holding longer must cast farther;
    a non-positive slope means noise, not signal — fall back to exploration)."""
    pts = [(float(h), float(d)) for h, d in samples if h is not None and d is not None]
    if len(pts) < min_samples:
        return None
    slopes = [(dj - di) / (hj - hi)
              for i, (hi, di) in enumerate(pts)
              for hj, dj in pts[i + 1:] if hj != hi]
    if not slopes:
        return None  # all holds identical — no slope recoverable
    slope = _median(slopes)
    if slope <= 0:
        return None
    intercept = _median([d - slope * h for h, d in pts])
    holds = [int(h) for h, _ in pts]
    return CastModel(slope, intercept, min(holds), max(holds), len(pts))


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def distance_for_hold(model: CastModel, hold_ms: int) -> float:
    """Predicted landing distance (px) for a hold duration."""
    return model.slope * hold_ms + model.intercept


def hold_for_distance(model: CastModel, distance_px: float) -> int:
    """Hold duration (ms) to land at `distance_px`, clamped to the model's
    observed hold support so the bot never extrapolates into untested
    charge times. Inverse of the linear fit."""
    raw = (distance_px - model.intercept) / model.slope
    return int(round(min(model.hold_max_ms, max(model.hold_min_ms, raw))))


def reachable(model: CastModel | None, distance_px: float, tol_px: float = 0.0) -> bool:
    """Whether the cast model can land within `tol_px` of `distance_px`.
    Always True when there's no model yet (exploration reaches by trying)."""
    if model is None:
        return True
    lo, hi = model.reach_px()
    return (lo - tol_px) <= distance_px <= (hi + tol_px)


def choose_target(
    fish: list[dict],
    model: CastModel | None,
    origin_x: int,
    tol_px: float = 0.0,
) -> dict | None:
    """Pick the fish to cast at.

    `fish` is the detector output: dicts with at least `x` and `kind`.
    Returns the chosen fish dict augmented with `target_dist` (px from
    `origin_x`) and `value`, or None when nothing is reachable.

    Policy (v1): highest point value among reachable fish, nearest as the
    tie-break. Reachability uses the cast model's support; with no model
    yet, every detected fish counts (the bot is exploring anyway).

    STRATEGY TODO (docs/fishing_minigame.md): a Whale is 5 points but
    catching it resets the streak to 1, throttling the climb back to
    Eel/Squid/Whale. Whether to skip Whales to keep a streak is an
    empirical call once outcomes are logged — left as pure value-max here.
    """
    best: dict | None = None
    best_key: tuple[int, float] | None = None
    for f in fish:
        dist = abs(f["x"] - origin_x)
        if not reachable(model, dist, tol_px):
            continue
        value = FISH_VALUE.get(f.get("kind", ""), 0)
        key = (value, -dist)  # max value, then nearest
        if best_key is None or key > best_key:
            best_key = key
            best = {**f, "target_dist": dist, "value": value}
    return best
