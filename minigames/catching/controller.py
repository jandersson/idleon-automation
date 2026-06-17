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
full derivation, and the dynamics fit verified against a dense trace
2026-06-17: g≈388 px/s², flap_vy≈-186 px/s, approach≈80 px/s):
  - Between flaps the avatar is in free fall: constant gravity `g` (px/s^2).
  - A flap is a set-velocity impulse: it sets vy to `flap_vy` (< 0 = upward),
    independent of the incoming velocity (standard Flappy physics).
  - The bob is a SYMMETRIC ballistic arc (~0.48s up, ~0.48s down, ~44px) — NOT
    the asymmetric "fast rise / slow descent" earlier notes claimed (that was
    an artifact of double-tap hover flaps).
  - Hoops scroll left at a constant `approach_speed` (px/s).

Firing rule — APEX-THREADING. The bob (~2*rise) is bigger than the passable
hole, so the avatar can only stay inside the hole through a crossing if it's
near its APEX (vy≈0, slowest) as the hoop passes — at mid-bob speed it moves
far more than the hole width. So fire the launch flap so the apex lands at the
crossing midpoint t_mid: when the hoop is one apex-time away (t_mid ≤
time_to_apex) and this flap's apex (fly_y - rise) lands inside the hole.
Between hoops main.py hovers with the apex on the hole centre, keeping the
avatar phase-ready and off the floor.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


# The detector's gap_top/gap_bottom is the ring's OUTER bounding box; the
# passable opening is narrower by the rim thickness. Inset the band by this
# before aiming the apex, so the apex targets the true hole, not a rim.
# Provisional from the wiki ring sprites (~24px wide rings); tune live if the
# avatar clips the top/bottom edge.
RING_INSET_PX = 12.0


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
    """Decide the phase-timed launch flap — APEX-THREADING.

    The bob (~2*rise ≈ 88px) is bigger than the passable hole, and at mid-bob
    speed the avatar moves far more than the hole width during a crossing. Only
    near the APEX (vy≈0) does it move slowly enough to stay inside the hole for
    the whole crossing (excursion = ½·g·(crossing/2)² ≈ 6px). So fire so the
    apex lands at the crossing midpoint: when the hoop is one apex-time away
    (t_mid ≤ time_to_apex) AND this flap's apex (fly_y - rise) would land
    inside the hole with room for the crossing excursion. False otherwise —
    main.py's apex-centred hover re-phases the avatar until a flap aligns.

    (descent_time_to / predict_descent_window remain for analysis; the firing
    rule deliberately does NOT cross mid-descent — that was the original bug.)
    """
    if gap_left_x is None or gap_right_x is None \
            or hole_top is None or hole_bottom is None:
        return False
    win = crossing_window(fly_x, gap_left_x, gap_right_x, dyn)
    if win is None:
        return False
    t_enter, t_exit = win
    if t_exit <= 0:
        return False  # hoop already crossed
    # A flap now reaches its apex time_to_apex from now. Fire only if that apex
    # lands WITHIN the crossing window (the avatar is at its slowest exactly as
    # the hoop passes) — bounded both ways, so it can't fire far-early (apex
    # before the hoop arrives) or far-late (apex after it's gone).
    t_apex = dyn.time_to_apex_s
    if not (t_enter <= t_apex <= t_exit):
        return False
    # ...and only if THIS flap's apex lands inside the PASSABLE opening (the
    # detected band inset by the rim thickness), with room for the small drift
    # over half the crossing. Else the avatar is at the wrong launch height;
    # main.py's apex-centred hover re-phases it.
    inner_top = hole_top + RING_INSET_PX
    inner_bottom = hole_bottom - RING_INSET_PX
    if inner_bottom <= inner_top:
        return False  # band too thin to commit a launch
    apex_y = fly_y - dyn.rise_height_px
    crossing = max(0.0, t_exit - t_enter)
    excursion = 0.5 * dyn.gravity * (crossing / 2.0) ** 2
    return inner_top + excursion <= apex_y <= inner_bottom - excursion


def hover_target_y(hole_center: float, dyn: Dynamics) -> float:
    """Between-hoop hover target for the model path: the bob BOTTOM (one rise
    below the hole centre) so the avatar bobs with its slow apex ON the hole
    centre — phase-ready for apex-threading, and never sinking toward the
    floor. Flap when fly_y drops to/below this."""
    return hole_center + dyn.rise_height_px


def floor_rescue_due(fly_y: float, fly_vy: float | None, play_height: float,
                     frac: float = 0.70, lookahead_s: float = 0.10) -> bool:
    """Hard floor backstop: flap if the avatar is below frac*play_height, or a
    fast descent is projected to cross that bound within lookahead_s. Catches
    the free-fall-to-floor sink (the 2026-06-17 trace run's death) that a
    position-only threshold misses at high descent speed — independent of the
    hover/coast/model state, so it guards every path. Pure kinematics, so it
    needs no fitted Dynamics and works on the hand-tuned path too."""
    bound = frac * play_height
    if fly_y >= bound:
        return True
    if fly_vy is not None and fly_vy > 0 and fly_y + fly_vy * lookahead_s >= bound:
        return True
    return False


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


def fit_flap_vy(rows: list[dict], gravity: float,
                min_apex_s: float = 0.3) -> float | None:
    """Flap impulse via the APEX-RISE method: for each clean flap, measure the
    rise = (y at the flap) - (min y reached before the next flap), then
    flap_vy = -sqrt(2*g*median_rise).

    More robust than differencing the launch velocity: at the ~37Hz / integer-
    pixel cadence the first post-flap finite difference is a poll-AVERAGED
    velocity, biased toward zero by ~g*dt/2 (~36 px/s) — it read -131 for a
    true ~-186 px/s (verified against this trace, 2026-06-17). The rise is a
    position measurement and carries no such bias. Skips truncated arcs (apex
    reached < min_apex_s after the flap — a rate-limited double-tap, not a full
    bob)."""
    flap_i = [i for i in range(len(rows)) if (_f(rows[i], "fired") or 0) >= 1.0]
    rises: list[float] = []
    for k, i in enumerate(flap_i):
        j = flap_i[k + 1] if k + 1 < len(flap_i) else len(rows)
        seg = [(_f(rows[s], "t"), _f(rows[s], "fly_y")) for s in range(i, j)]
        seg = [(t, y) for t, y in seg if t is not None and y is not None]
        if len(seg) < 3:
            continue
        t0, y0 = seg[0]
        ymin = min(y for _, y in seg)
        t_apex = next(t for t, y in seg if y == ymin) - t0
        if t_apex < min_apex_s:
            continue  # truncated double-tap arc, not a full bob
        if y0 - ymin > 0:
            rises.append(y0 - ymin)
    rise = _median(rises)
    if rise is None or rise <= 0:
        return None
    return -math.sqrt(2.0 * gravity * rise)


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
    the caller keeps the hand-tuned fallback rather than fly on a bad fit.
    flap_vy is fit AFTER gravity (the apex-rise method needs g)."""
    g = fit_gravity(rows)
    ap = fit_approach_speed(rows)
    if g is None or ap is None:
        return None
    fv = fit_flap_vy(rows, gravity=g)
    if fv is None:
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
