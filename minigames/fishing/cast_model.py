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


# --- Moving-target lead --------------------------------------------------
# The targets SLIDE back and forth along the bar, starting ~cast 5 and growing
# in speed/amplitude through a game (measured: ~±15-22 px/s late-game; see
# docs/fishing_minigame.md). The bot otherwise aims at the fish's decision-time
# x, but over the ~1.3s from decision to landing the fish slides 14-32px away —
# the dominant miss cause that caps the streak at ~3. These pure helpers lead
# the aim to the fish's PREDICTED landing position.
#
# SHIPPED LOGGING-FIRST as a parity A/B (aim_mode 'model_lead' vs 'model_nolead')
# — the effective catch radius is only ~10px and the cast model's own error has
# a heavy tail, so a lead with a few-px error budget can be NET-NEGATIVE. Promote
# off the A/B only on a disjoint-CI make-rate win under matched velocity bins
# (the #23/#38 discipline). Two safeguards earn their keep below.
FLIGHT_S = 0.7              # lure flight (decision-anchor -> landing); constant first guess
LEAD_TIME_FALLBACK_S = 1.3  # decision -> landing time (hold ~0.6 + flight ~0.7) until hold_ms is fit
# Below this the apparent velocity is at the integer-centroid noise floor (a
# ±1px jitter over ~0.12s aliases to ~±8px/s) or the fish is near a turn — either
# way, don't lead.
MIN_LEAD_VEL_PX_S = 8.0
# Turn-cap: the slide is monotonic only ~0.8s but the lead spans ~1.3s, so a
# reversal can fall inside the flight. Bounding the lead to ~one catch radius
# keeps an over-lead-past-a-turn from converting a make into a miss.
MAX_LEAD_PX = 12.0
MIN_LEAD_SAMPLES = 3        # fewer can't beat the ±1px centroid jitter
MIN_LEAD_SPAN_S = 0.12      # too short a baseline can't resolve sub-pixel motion


def theil_sen_velocity(samples: list[tuple[float, float]],
                       min_samples: int = MIN_LEAD_SAMPLES,
                       min_span_s: float = MIN_LEAD_SPAN_S) -> float | None:
    """Robust x-velocity (px/s) from ``[(t, x), ...]`` fish sightings, or None
    when unresolved. Theil-Sen (median of pairwise slopes, like fit_cast_model)
    beats the ±1px integer-centroid jitter that aliases a 2-frame delta to
    ±20px/s. Returns None below ``min_samples``, when the time span is too short
    to resolve sub-pixel motion, or when consecutive samples REVERSE direction (a
    turn / pure noise → don't lead). Pure."""
    pts = [(float(t), float(x)) for t, x in samples]
    if len(pts) < min_samples:
        return None
    if pts[-1][0] - pts[0][0] < min_span_s:
        return None
    deltas = [pts[i + 1][1] - pts[i][1] for i in range(len(pts) - 1)]
    if any(d > 0 for d in deltas) and any(d < 0 for d in deltas):
        return None  # direction reversed across the window -> at a turn / noisy
    slopes = [(xj - xi) / (tj - ti)
              for i, (ti, xi) in enumerate(pts)
              for tj, xj in pts[i + 1:] if tj != ti]
    return _median(slopes) if slopes else None


def lead_fish_dist(target_dist_px: float, velocity_px_s: float | None,
                   lead_time_s: float, reach_lo: float, reach_hi: float,
                   min_vel_px_s: float = MIN_LEAD_VEL_PX_S,
                   max_lead_px: float = MAX_LEAD_PX) -> tuple[float, float, bool]:
    """Lead a sliding fish: the target distance (px from the cast origin) adjusted
    to the fish's PREDICTED position at landing. Returns
    ``(led_dist, effective_lead_px, clamped)``.

    No lead (``led == target``) when ``velocity`` is None or below the noise/turn
    floor. The lead is turn-capped to ``±max_lead_px`` and reach-clamped to
    ``[reach_lo, reach_hi]`` — the cast model would silently swallow an
    out-of-reach lead, so ``clamped`` flags it. Operates on the scalar distance
    from the FIXED left-edge origin, so a +velocity (fish sliding outward/right)
    increases the distance and −velocity decreases it; there's no abs() direction
    flip. Pure."""
    if velocity_px_s is None or abs(velocity_px_s) < min_vel_px_s:
        return target_dist_px, 0.0, False
    lead = max(-max_lead_px, min(max_lead_px, velocity_px_s * lead_time_s))
    led = target_dist_px + lead
    clamped_led = max(reach_lo, min(reach_hi, led))
    return clamped_led, clamped_led - target_dist_px, clamped_led != led
