"""Predictive flap timer for the catching (Flappy-style) bot (#60).

Replaces the hand-picked phase-timing constants (LEAD_DIST_PX, COAST_S,
TIMED_LOW_OFFSET_PX in main.py) with a model fitted from a dense per-poll
trajectory trace (main.py --trace). The avatar's bob is bigger than a ring's
passable hole, so it can't hover THROUGH a ring — it has to coast through,
timed so its slow descent lines up with the crossing. This module decides
WHEN to fire that launch flap from the fitted dynamics.

Pure / IO-free and unit-tested (tests/test_catching_controller.py); main.py
owns the capture/click loop and the hover + coast-rescue fallback.

Physics model (screen y grows downward; see docs/catching_timing.md for the
full derivation):
  - Between flaps the avatar is in free fall: constant gravity `g` (px/s^2).
  - A flap is a set-velocity impulse: it sets vy to `flap_vy` (< 0 = upward),
    independent of the incoming velocity (standard Flappy physics).
  - Hoops scroll left at a constant `approach_speed` (px/s).

Firing rule — fire the launch flap when the hoop has approached close enough
that flapping NOW lands the avatar's slow descent at the hole as the hoop
crosses. Formally: let T be the time for a flap-now to descend to the hole
centre, and t_mid the crossing midpoint (when the hoop's horizontal span is
centred on the avatar's fixed x). T is ~constant; t_mid shrinks as the hoop
nears. Fire at the crossover t_mid <= T, so the descent-centre arrival aligns
with the crossing. The descent is slow, so the avatar lingers in the hole
across the brief crossing.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dynamics:
    """Fitted vertical dynamics + hoop approach speed.

    gravity: px/s^2, > 0 (y grows downward).
    flap_vy: px/s, vy immediately after a flap; < 0 (upward).
    approach_speed: px/s, > 0 (hoops scroll left at this rate).
    """
    gravity: float
    flap_vy: float
    approach_speed: float

    @property
    def rise_height_px(self) -> float:
        """How far a single flap lifts the avatar above its launch point —
        flap_vy^2 / (2 g). The bob's effective amplitude."""
        return self.flap_vy ** 2 / (2.0 * self.gravity)

    @property
    def time_to_apex_s(self) -> float:
        """Time from a flap to the top of the arc — |flap_vy| / g."""
        return abs(self.flap_vy) / self.gravity

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Dynamics":
        return cls(gravity=float(d["gravity"]), flap_vy=float(d["flap_vy"]),
                   approach_speed=float(d["approach_speed"]))


def step(y: float, vy: float, dt: float, dyn: Dynamics) -> tuple[float, float]:
    """Advance a free-falling avatar by dt; return (y, vy)."""
    return (y + vy * dt + 0.5 * dyn.gravity * dt * dt, vy + dyn.gravity * dt)


def descent_time_to(y0: float, target_y: float, dyn: Dynamics,
                    vy0: float | None = None) -> float | None:
    """Time for a flap-launched trajectory from y0 to reach target_y on the
    DESCENT (the larger root of the position quadratic). vy0 defaults to a
    fresh flap (dyn.flap_vy). Returns None when target_y is above the apex —
    the trajectory never descends to it (too weak a flap / avatar too low)."""
    vy0 = dyn.flap_vy if vy0 is None else vy0
    g = dyn.gravity
    # 0.5 g t^2 + vy0 t + (y0 - target_y) = 0; descending root uses +sqrt.
    a, b, c = 0.5 * g, vy0, (y0 - target_y)
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    t = (-b + math.sqrt(disc)) / (2 * a)
    return t if t > 0 else None


def predict_descent_window(y0: float, hole_top: float, hole_bottom: float,
                           dyn: Dynamics) -> tuple[float, float, float] | None:
    """For a flap fired now from y0, return (t_apex, t_in, t_out): the time to
    the top of the arc, and the descent times entering (hole_top) and leaving
    (hole_bottom) the hole. None when the flap can't carry the descent into the
    hole (apex below hole_top). Mainly for analysis/tests; the firing decision
    uses descent_time_to directly."""
    t_apex = dyn.time_to_apex_s
    t_in = descent_time_to(y0, hole_top, dyn)
    t_out = descent_time_to(y0, hole_bottom, dyn)
    if t_in is None or t_out is None:
        return None
    return (t_apex, t_in, t_out)


def crossing_window(fly_x: float, gap_left_x: float, gap_right_x: float,
                    dyn: Dynamics) -> tuple[float, float] | None:
    """When the hoop's horizontal span covers the avatar's fixed x: (t_enter,
    t_exit) seconds from now. The avatar must be in the hole across this
    interval to thread. None when approach_speed is unset."""
    if dyn.approach_speed <= 0:
        return None
    t_enter = (gap_left_x - fly_x) / dyn.approach_speed
    t_exit = (gap_right_x - fly_x) / dyn.approach_speed
    return (t_enter, t_exit)


def should_flap_now(fly_x: float, fly_y: float,
                    gap_left_x: float | None, gap_right_x: float | None,
                    hole_top: float | None, hole_bottom: float | None,
                    dyn: Dynamics) -> bool:
    """Decide the phase-timed launch flap. True when the hoop has approached
    close enough that flapping now drops the avatar's slow descent through the
    hole centre as the hoop crosses (t_mid <= descent-to-centre time). False
    with no hoop, an unfit approach speed, a hoop already past, or a flap that
    can't reach the hole on descent — main.py's hover/coast handles those."""
    if gap_left_x is None or gap_right_x is None \
            or hole_top is None or hole_bottom is None:
        return False
    win = crossing_window(fly_x, gap_left_x, gap_right_x, dyn)
    if win is None:
        return False
    t_enter, t_exit = win
    if t_exit <= 0:
        return False  # hoop already crossed
    t_mid = 0.5 * (t_enter + t_exit)
    hole_center = 0.5 * (hole_top + hole_bottom)
    t_descent = descent_time_to(fly_y, hole_center, dyn)
    if t_descent is None:
        return False  # a flap now can't carry the descent into the hole
    return t_mid <= t_descent


# --- Fitting the dynamics from a dense --trace CSV -------------------------
#
# These recover (gravity, flap_vy, approach_speed) from a list of per-poll
# trace rows (TRACE_COLUMNS in main.py). Heuristic but robust to the trace's
# noise (integer pixel detections, ~60% hoop-detection rate, finite-difference
# velocities); refine the segment thresholds against a real trace. A row is a
# dict; numeric fields may be None/"" where the detector had no value.

def _f(row: dict, key: str) -> float | None:
    """Coerce a trace cell to float, or None for missing/blank."""
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fit_quadratic_accel(ts: list[float], ys: list[float]) -> float | None:
    """Least-squares gravity from y(t) = c0 + c1 t + 0.5 g t^2 over one
    free-fall run (>= 3 points). Returns g (the 2*quadratic-coeff) or None."""
    n = len(ts)
    if n < 3:
        return None
    # Normal equations for [1, t, t^2], solved with a tiny Gaussian
    # elimination — keeps this module numpy-free and importable anywhere.
    S = [[0.0] * 4 for _ in range(3)]  # augmented [A | b]
    for t, y in zip(ts, ys):
        powers = [1.0, t, t * t]
        for i in range(3):
            for j in range(3):
                S[i][j] += powers[i] * powers[j]
            S[i][3] += powers[i] * y
    coeffs = _solve3(S)
    if coeffs is None:
        return None
    g = 2.0 * coeffs[2]
    return g if g > 0 else None


def _solve3(aug: list[list[float]]) -> list[float] | None:
    """Solve a 3x3 linear system given as an augmented matrix; None if
    singular."""
    m = [row[:] for row in aug]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-9:
            return None
        m[col], m[piv] = m[piv], m[col]
        pivot = m[col][col]
        m[col] = [v / pivot for v in m[col]]
        for r in range(3):
            if r == col:
                continue
            factor = m[r][col]
            m[r] = [v - factor * mc for v, mc in zip(m[r], m[col])]
    return [m[0][3], m[1][3], m[2][3]]


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def fit_gravity(rows: list[dict], min_run: int = 4) -> float | None:
    """Median gravity over free-fall runs — maximal stretches of consecutive
    polls with no flap fired, fitting y(t) per run."""
    gs: list[float] = []
    run_t: list[float] = []
    run_y: list[float] = []

    def flush():
        if len(run_t) >= min_run:
            g = _fit_quadratic_accel(run_t, run_y)
            if g is not None:
                gs.append(g)

    for row in rows:
        t, y, fired = _f(row, "t"), _f(row, "fly_y"), _f(row, "fired")
        if t is None or y is None:
            flush(); run_t.clear(); run_y.clear(); continue
        if fired and fired >= 1.0:
            flush(); run_t.clear(); run_y.clear()
            continue  # the flap poll ends the free-fall run
        run_t.append(t); run_y.append(y)
    flush()
    return _median(gs)


def fit_flap_vy(rows: list[dict], window_s: float = 0.25) -> float | None:
    """Median peak upward velocity after a flap — the most-negative fly_vy in
    the window_s following each fired poll. The launch impulse the model sets
    vy to."""
    peaks: list[float] = []
    for i, row in enumerate(rows):
        if not (_f(row, "fired") or 0) >= 1.0:
            continue
        t0 = _f(row, "t")
        if t0 is None:
            continue
        best = None
        for row2 in rows[i:]:
            t = _f(row2, "t")
            if t is None or t - t0 > window_s:
                break
            vy = _f(row2, "fly_vy")
            if vy is not None and (best is None or vy < best):
                best = vy
        if best is not None and best < 0:
            peaks.append(best)
    return _median(peaks)


def fit_approach_speed(rows: list[dict], min_run: int = 4) -> float | None:
    """Median hoop scroll speed (px/s, positive) from within-hoop runs where
    gap_left_x decreases monotonically — regress gap_left_x on t per run and
    take -slope. Runs break when the detector drops the hoop or switches to the
    next one (gap_left_x jumps up)."""
    speeds: list[float] = []
    run_t: list[float] = []
    run_x: list[float] = []

    def flush():
        if len(run_t) >= min_run:
            slope = _linregress_slope(run_t, run_x)
            if slope is not None and slope < 0:
                speeds.append(-slope)

    prev_x = None
    for row in rows:
        t, x = _f(row, "t"), _f(row, "gap_left_x")
        if t is None or x is None:
            flush(); run_t.clear(); run_x.clear(); prev_x = None; continue
        if prev_x is not None and x > prev_x + 2:
            # gap jumped right — a new hoop became "next"; start a fresh run.
            flush(); run_t.clear(); run_x.clear()
        run_t.append(t); run_x.append(x); prev_x = x
    flush()
    return _median(speeds)


def _linregress_slope(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den < 1e-9:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def fit_dynamics(rows: list[dict]) -> Dynamics | None:
    """Fit a Dynamics from a list of trace rows (one per poll). Returns None
    when any of the three parameters can't be recovered (too little data) —
    the caller keeps the hand-tuned fallback rather than fly on a bad fit."""
    g = fit_gravity(rows)
    fv = fit_flap_vy(rows)
    ap = fit_approach_speed(rows)
    if g is None or fv is None or ap is None:
        return None
    return Dynamics(gravity=g, flap_vy=fv, approach_speed=ap)


def load_trace(path: Path) -> list[dict]:
    """Read a --trace CSV into a list of row dicts (string cells)."""
    import csv
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_dynamics(path: Path) -> Dynamics | None:
    """Load a fitted Dynamics from JSON, or None if the file is absent."""
    if not path.exists():
        return None
    return Dynamics.from_dict(json.loads(path.read_text()))


def save_dynamics(path: Path, dyn: Dynamics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dyn.to_dict(), indent=2))
