"""Fishing-minigame planning logic — pure, no Tk / no IO.

The fishing minigame is a HOLD-to-charge distance game (wiki: Fishing
Minigame): hold the mouse to fill a charge bar, release to cast the lure a
distance set by the charge LEVEL (the bar's fill height), landing it on a
fish (Green 1 / Eel 2 / Squid 3 / Whale 5 points) while avoiding mines.

The bot casts CLOSED-LOOP: it polls the charge bar while holding and
releases at a target fill (common.input.charge_and_release), so the charge
level — not the hold duration — is the control variable, and the core
learned model is charge_level <-> cast distance.

Why charge, not hold (see docs/fishing_minigame.md "Decision record"):
offline on the saved run-7 frames, landing distance is ~linear in the bar
fill (landed_x ~ 5.0 * charge, R^2 ~ 0.999) and saturates once the bar is
full (charge ~ 60). The charge fill is in-crop and stable after release; the
earlier hold_ms -> distance model fought BOTH the off-crop bobber landing
noise AND hold->charge drift, and undershot near the cap. charge -> distance
sidesteps both.

This module holds the two pure pieces:

1. `CastModel` + `fit_cast_model` — a robust (Theil-Sen) charge_level ->
   landing-distance fit over logged (charge_level, landed_dist_px) casts,
   plus the inverse `charge_for_distance` the bot uses to aim. The distance
   CAP is encoded by the charge support: the bar can't fill past charge_max,
   so the model never aims beyond slope*charge_max+intercept (reach_px()) —
   no separate piecewise-flat term needed. Mirrors the darts stripe_model:
   fitted once at startup from the DB, exploration casts feed the next fit.

2. `choose_target` — pick which fish to cast at: highest point value that
   the cast model can actually reach, nearest as the tie-break.

CALIBRATION: `fit_cast_model` returns None until MIN_SAMPLES clean casts
exist, so the bot explores (random target charge levels, logged) until the
relationship is pinned. See docs/fishing_minigame.md.
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

# Casts needed before the charge fit is trusted over exploration. The
# relationship is 1-D and smooth, so this is modest; raise if observe data
# is noisy.
MIN_SAMPLES = 12


@dataclass
class CastModel:
    """Linear charge_level -> landing-distance fit (distance in px from the
    cast origin). `charge_min`/`charge_max` bound the observed support; the
    bot never targets a charge outside it — the bar physically caps the fill,
    and the fit can't be trusted where no cast has been sampled (the darts
    pose-support lesson). So the distance cap falls straight out of the charge
    support."""
    slope: float       # px gained per unit of charge fill
    intercept: float   # px at charge = 0
    charge_min: int
    charge_max: int
    n: int

    def reach_px(self) -> tuple[float, float]:
        """(min, max) landing distance the model can aim for, i.e. the
        distances at the support's charge bounds."""
        return (
            self.slope * self.charge_min + self.intercept,
            self.slope * self.charge_max + self.intercept,
        )


def fit_cast_model(
    samples: list[tuple[int, float]],
    min_samples: int = MIN_SAMPLES,
) -> CastModel | None:
    """Robust (Theil-Sen) line through (charge_level, landed_dist_px) samples.

    Charge is the clean predictor (the bar fill is stable + in-crop), but the
    bobber landing (the y) still carries noise — an occasional off-crop or
    mis-latched landing gives an outlier. Least-squares would let those skew the
    aim, so the slope is the MEDIAN of all pairwise slopes (Theil-Sen, robust to
    ~29% outliers) and the intercept the median residual.

    Returns None below `min_samples`, when the charges don't vary (degenerate),
    or when the robust slope is non-positive (more charge must cast farther; a
    non-positive slope means noise, not signal — fall back to exploration)."""
    pts = [(float(c), float(d)) for c, d in samples if c is not None and d is not None]
    if len(pts) < min_samples:
        return None
    slopes = [(dj - di) / (cj - ci)
              for i, (ci, di) in enumerate(pts)
              for cj, dj in pts[i + 1:] if cj != ci]
    if not slopes:
        return None  # all charges identical — no slope recoverable
    slope = _median(slopes)
    if slope <= 0:
        return None
    intercept = _median([d - slope * c for c, d in pts])
    charges = [int(c) for c, _ in pts]
    return CastModel(slope, intercept, min(charges), max(charges), len(pts))


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def distance_for_charge(model: CastModel, charge: float) -> float:
    """Predicted landing distance (px) for a charge-bar fill level."""
    return model.slope * charge + model.intercept


def charge_for_distance(model: CastModel, distance_px: float) -> int:
    """Charge-bar fill level to land at `distance_px`, clamped to the model's
    observed charge support so the bot never extrapolates past the bar's
    saturation or into untested fills. Inverse of the linear fit; the result
    is the target the closed-loop cast releases at."""
    raw = (distance_px - model.intercept) / model.slope
    return int(round(min(model.charge_max, max(model.charge_min, raw))))


def reachable(model: CastModel | None, distance_px: float, tol_px: float = 0.0) -> bool:
    """Whether the cast model can land within `tol_px` of `distance_px`.
    Always True when there's no model yet (exploration reaches by trying)."""
    if model is None:
        return True
    lo, hi = model.reach_px()
    return (lo - tol_px) <= distance_px <= (hi + tol_px)


def lands_on_mine_only(landing_x: float, mines: list[dict], fish: list[dict],
                       tol_px: float = 15.0) -> bool:
    """Whether a cast landing at play-region x `landing_x` hits a MINE with no
    fish there — the only way to fail (wiki: landing on a fish counts even with a
    mine under it; only a mine-only spot fails). `mines`/`fish` are detector
    dicts with 'x'. Used to keep exploratory casts off mine-only spots (a cast
    that targets a fish lands on the fish, so it's already safe)."""
    if not any(abs(m["x"] - landing_x) <= tol_px for m in mines):
        return False
    return not any(abs(f["x"] - landing_x) <= tol_px for f in fish)


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
