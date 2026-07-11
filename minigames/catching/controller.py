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


def coast_rescue_due(fly_y: float, fly_vy: float | None,
                     floor_y: float) -> bool:
    """Mid-coast rescue: flap only if the avatar is BELOW the coast floor AND
    measurably DESCENDING. The descent gate is load-bearing: the launch flap
    starts 33-55px below the hole centre — deeper than the coast floor
    (gb - COAST_RESCUE_PX) — so an ascent-blind rescue re-flaps the fresh
    launch on every rate-limit tick, pinning vy at flap_vy until the avatar
    has risen past the floor, and the last flap's arc overshoots the apex
    ~24px above the hole centre into the TOP rim (#62's sim replay; run 16's
    live death had the same over-lift signature). A None vy (detection-gap
    frame) does NOT rescue — floor_rescue_due's position bound still
    backstops a genuine sink."""
    return fly_vy is not None and fly_vy > 0 and fly_y > floor_y


def floor_rescue_due(fly_y: float, fly_vy: float | None, play_height: float,
                     frac: float = 0.70, lookahead_s: float = 0.05) -> bool:
    """Hard floor backstop: flap if the avatar is below frac*play_height, or a
    fast descent is projected to cross that bound within lookahead_s. Catches
    the free-fall-to-floor sink (the 2026-06-17 trace run's death) that a
    position-only threshold misses at high descent speed — independent of the
    hover/coast/model state, so it guards every path. Pure kinematics, so it
    needs no fitted Dynamics and works on the hand-tuned path too.

    lookahead_s is short (~1-2 polls at ~37Hz): a longer horizon fired it ~18px
    early (run 16), preempting the model launch that fires in the same low zone.
    Detection-gap falls are caught by the position bound, not the lookahead."""
    bound = frac * play_height
    if fly_y >= bound:
        return True
    if fly_vy is not None and fly_vy > 0 and fly_y + fly_vy * lookahead_s >= bound:
        return True
    return False


# --- Phase-locked flap planner (#61) ---------------------------------------
#
# plan_flap upgrades the model path from "hover and hope the launch window
# coincides" to an explicit per-ring plan. Everything is per-poll and
# stateless (main.py's loop and sim.py call it identically); the only state
# the caller carries is a SpeedTracker for the live scroll speed. Grounded
# in the #62 measurements (docs/catching_timing.md):
#
#   - The in-game flap lands ~ACT_LATENCY_S after the fire decision, so
#     every prediction projects the avatar forward by that lead.
#   - A flap may fire at ANY point of the descent, which makes apex height
#     and apex time jointly controllable across two bobs: flapping deeper
#     brings the next pass of the launch anchor EARLIER (lower apex, short
#     fall-back), flapping higher pushes it LATER (tall bob). The SETUP rule
#     picks the flap moment on the current descent that lands the next
#     descent on the anchor y* = hole_centre + rise exactly at the launch
#     decision time t* = t_mid - time_to_apex - ACT_LATENCY_S.
#   - Rings are small ovals: passing entirely above/below the outer band is
#     a safe DODGE (no score, no death). When the timing is unsalvageable or
#     the anchor is unreachable (deep rings), the planner holds the bob
#     clear of the band instead of gambling a mid-bob crossing.
#   - The scroll speed RAMPS (~63 -> ~105 px/s over a run), so timing uses a
#     live speed estimate (SpeedTracker), not the fitted constant.

ACT_LATENCY_S = 0.075   # decision -> in-game flap (measured, #62)
LAUNCH_TOL_S = 0.12     # |apex time - crossing midpoint| to fire the launch
SETUP_TOL_S = 0.05      # setup fires when the solve crosses within this
AVATAR_HALF_W = 5.0     # avatar half-width for the crossing-window length
DODGE_MARGIN_PX = 4.0   # clearance beyond the band edge for a dodge bob
CEILING_MARGIN_PX = 10.0  # over-dodge apex must stay below the crop top
# A trigger-crossing flap keeps falling through the actuation latency plus
# up to a poll of decision delay before the impulse lands (~300 px/s * 0.1s
# and change) — every depth bound must leave this much room above the
# lethal floor.
LATENCY_FALL_PX = 34.0
FLOOR_BELOW_PLAN_PX = 20.0  # emergency floor sits this far below the plan
FLOOR_FRAC_CAP = 0.80       # ...but never deeper than this fraction
# Callers must pass fly_vy=None for this long after any fired click: the
# finite-difference velocity spans the ~160ms click stall (the flap's turn
# happens INSIDE the gap), so it reads near-zero garbage — firing a launch
# on it is how echo double-flaps sneak back in wearing a 'launch' tag.
VY_BLACKOUT_S = 0.25
# ...and suppress non-launch plan flaps for this long after a fired click:
# right after the stall the avatar is still near its (deep) flap point, so
# trigger-crossing conditions re-read as true and re-flap the fresh bob —
# the echo. By 0.3s a landed flap has clearly risen clear of the triggers.
ECHO_SUPPRESS_S = 0.30


class SpeedTracker:
    """Live hoop scroll speed from per-poll gap_left_x samples.

    The scroll speed ramps through a run, so the fitted approach_speed goes
    stale — feed every detected gap_left_x in and read speed() for the
    freshest estimate. Samples jumping RIGHT (a new hoop became "next")
    reset the window. Returns None until enough samples span the window."""

    def __init__(self, window_s: float = 1.4, min_samples: int = 4):
        self.window_s = window_s
        self.min_samples = min_samples
        self._pts: list[tuple[float, float]] = []

    def add(self, t: float, gap_left_x: float) -> None:
        if self._pts and gap_left_x > self._pts[-1][1] + 8:
            self._pts.clear()   # next hoop became the target
        self._pts.append((t, gap_left_x))
        cutoff = t - self.window_s
        self._pts = [(pt, px) for pt, px in self._pts if pt >= cutoff]

    def speed(self) -> float | None:
        """Scroll speed in px/s (positive = leftward), or None."""
        if len(self._pts) < self.min_samples:
            return None
        slope = _linregress_slope([p[0] for p in self._pts],
                                  [p[1] for p in self._pts])
        if slope is None or slope > -20:
            return None
        return -slope


class RingProjector:
    """Per-poll ring geometry with dropout bridging.

    The detector reports the next ring on only ~67% of polls; the planner
    needs geometry (and timing!) EVERY poll, and a stale gap_left_x is a
    timing error of speed*staleness. Feed detections in; get() projects the
    last detection forward at the tracked scroll speed (vertical extent is
    static per ring). Returns None once the projection is stale. Also owns
    the SpeedTracker — one object for the callers to carry."""

    def __init__(self, fly_x: float, fallback_speed: float,
                 max_stale_s: float = 0.5):
        self.tracker = SpeedTracker()
        self.fly_x = fly_x
        self.fallback_speed = fallback_speed
        self.max_stale_s = max_stale_s
        self._last: tuple[float, float, float, float, float] | None = None

    def update(self, t: float, gl: float, gr: float,
               gt: float, gb: float) -> None:
        # The detector switches to the NEXT ring once the current ring's
        # centre passes the fly — while the old ring still x-overlaps the
        # avatar. Planning against the new ring there can flap through the
        # old ring's rim, so hold the old projection until it fully clears.
        if self._last is not None and gl > self._last[1] + 8:
            old = self.get(t)
            if old is not None and old[1] + AVATAR_HALF_W >= self.fly_x:
                return
        self.tracker.add(t, gl)
        self._last = (t, gl, gr, gt, gb)

    def speed(self) -> float:
        return self.tracker.speed() or self.fallback_speed

    def get(self, t: float) -> tuple[float, float, float, float] | None:
        """Projected (gl, gr, gt, gb) at time t, or None if stale/empty."""
        if self._last is None:
            return None
        t0, gl, gr, gt, gb = self._last
        if t - t0 > self.max_stale_s:
            return None
        d = self.speed() * (t - t0)
        return (gl - d, gr - d, gt, gb)


def _fall_time(dy: float, gravity: float) -> float:
    """Free-fall time from rest over dy px (>=0)."""
    return math.sqrt(2.0 * max(0.0, dy) / gravity)


def plan_flap(fly_y: float, fly_vy: float | None,
              gap_left_x: float | None, gap_right_x: float | None,
              hole_top: float | None, hole_bottom: float | None,
              fly_x: float, play_height: float, dyn: Dynamics,
              speed: float | None = None) -> tuple[bool, str, float | None]:
    """Per-poll phase-locked flap decision for the next ring.

    Returns (flap_now, mode, floor_frac_override):
      mode: 'launch'  — the apex of a flap fired now lands in the hole at
                        the crossing; the caller starts the coast.
            'setup'   — bob-restart flap timed so the NEXT descent reaches
                        the launch anchor at launch time.
            'hover'   — natural anchor-cycle flap (ring far, or no better
                        plan); also the no-ring fallback.
            'dodge_under' / 'dodge_over' — hold the bob clear of the band
                        through an unthreadable crossing.
            'wait'    — no flap this poll.
      floor_frac_override: pass to floor_rescue_due's frac while an
      under-dodge needs the bob deeper than the default bound (else None).

    Never fires while ascending (measured fly_vy < 0): re-flapping a rising
    avatar is the echo/over-lift failure. A None fly_vy plans as if at rest
    for hover/dodge triggers but never fires a launch (the apex prediction
    would be unreliable at the moment precision matters most)."""
    rise = dyn.rise_height_px
    t_apex = dyn.time_to_apex_s
    g = dyn.gravity

    ascending = fly_vy is not None and fly_vy < 0
    vy = max(0.0, fly_vy) if fly_vy is not None else 0.0

    have_ring = (gap_left_x is not None and gap_right_x is not None
                 and hole_top is not None and hole_bottom is not None)
    if not have_ring:
        # No ring in sight: natural cycle on the caller's held/default
        # target is the caller's job; just never flap while ascending.
        return (False, "wait", None)

    hc = 0.5 * (hole_top + hole_bottom)
    v = speed if speed and speed > 0 else dyn.approach_speed
    if v <= 0:
        return (False, "wait", None)

    center_x = 0.5 * (gap_left_x + gap_right_x)
    half_w = 0.5 * (gap_right_x - gap_left_x) + AVATAR_HALF_W
    t_mid = (center_x - fly_x) / v
    t_half = half_w / v
    if t_mid + t_half < 0:
        return (False, "wait", None)   # ring fully past

    anchor_y = hc + rise                     # flap here -> apex on hc
    t_star = t_mid - t_apex - ACT_LATENCY_S  # ideal launch-decision time
    floor_cap = play_height - LATENCY_FALL_PX - 4.0
    # While a ring is being worked, planned flaps (setup especially) go
    # legitimately as deep as floor_cap — the emergency floor must sit at
    # its full depth or its lookahead preempts/echoes every planned flap
    # with its own timing, un-phasing the plan. The caller additionally
    # echo-suppresses the floor path (see main.py/sim.py).
    work_y = min(anchor_y, floor_cap)
    plan_frac = FLOOR_FRAC_CAP

    # --- LAUNCH: fire now if this flap threads the WHOLE crossing ---------
    # The avatar rides y(t) = apex + g/2 (t - t_apex)^2 through the overlap
    # window [t_mid - t_half, t_mid + t_half]. It must stay inside the
    # passable hole for the entire window: the apex (highest point) above
    # the hole top, and the deepest in-window point — at the window edge
    # farthest from the apex — above the hole bottom. A static apex-time
    # tolerance is NOT equivalent: an apex accepted 0.12s early has fallen
    # g/2*(0.12+t_half)^2 ~ 19px by the window's far edge, straight through
    # the hole bottom into the rim (the recalibrated sim's edge-clip
    # signature).
    if not ascending and fly_vy is not None:
        y_land = fly_y + vy * ACT_LATENCY_S + 0.5 * g * ACT_LATENCY_S ** 2
        apex_y = y_land - rise
        inner_top = hole_top + RING_INSET_PX
        inner_bottom = hole_bottom - RING_INSET_PX
        apex_dt = ACT_LATENCY_S + t_apex
        worst_dt = max(abs((t_mid - t_half) - apex_dt),
                       abs((t_mid + t_half) - apex_dt))
        deepest_y = apex_y + 0.5 * g * worst_dt ** 2
        if (abs(apex_dt - t_mid) <= LAUNCH_TOL_S
                and apex_y >= inner_top and deepest_y <= inner_bottom):
            return (True, "launch", plan_frac)

    # A flap fired within ~a full bob (incl. latency) of the window start
    # is still in flight DURING the crossing — there is no neutral flap in
    # that zone. The launch and setup arcs are window-shaped/band-safe by
    # construction; everything else must be a side-dodge or nothing. A
    # plain anchor-cycle flap there parks its apex INSIDE the band
    # mid-window (the deterministic ring-1 death the sim exposed).
    committed = (t_mid - t_half) <= (ACT_LATENCY_S + 2.0 * t_apex + 0.1)

    # Predicted flap LANDING point: the avatar keeps falling through the
    # actuation latency, so every trigger fires on where the impulse will
    # actually land, not on raw position. Unknown vy projects as 0 — the
    # flap fires late and lands deeper, which is the safe direction for
    # every trigger here (deeper = away from the band, bounded by the
    # emergency floor).
    y_land_t = fly_y + vy * ACT_LATENCY_S + 0.5 * g * ACT_LATENCY_S ** 2

    def _side_dodge():
        """Dodge on the side the avatar is already on — no time to cross
        the band. Fire when the predicted LANDING reaches the target so
        the bob's apex clears the band edge regardless of descent speed
        (a fixed trigger credit under-corrects slow descents and pokes
        the apex into the band)."""
        under_target = hole_bottom + DODGE_MARGIN_PX + rise
        over_target = hole_top - DODGE_MARGIN_PX
        under_ok = under_target <= floor_cap + LATENCY_FALL_PX
        over_ok = (over_target - rise >= CEILING_MARGIN_PX
                   and fly_y <= over_target + 8.0)
        if over_ok and not (under_ok and fly_y > hc):
            return (not ascending and y_land_t >= over_target,
                    "dodge_over", None)
        if under_ok:
            # Landing target below the band; deeper than the anchor, so an
            # aligned launch arc never hits it first. The caller must run
            # the emergency floor POSITIONAL-ONLY in this mode (vy=None):
            # the lookahead fires ~15px above the bound, which is exactly
            # the bob's band clearance.
            return (not ascending and y_land_t >= under_target,
                    "dodge_under", plan_frac)
        # Deep band (no dodge fits): hold the clamped anchor and hope the
        # natural apex catches the hole.
        return (not ascending and y_land_t >= work_y, "hover", plan_frac)

    # --- launch window missed entirely: get clear of the band -------------
    if t_star <= SETUP_TOL_S:
        return _side_dodge()

    # --- SETUP: land the next descent on the anchor at launch time --------
    # Restarting the bob now passes the anchor (descending) at
    #   T = ACT_LATENCY + t_apex + fall_time(anchor - apex)
    # T shrinks as the flap is delayed (deeper flap -> lower apex -> shorter
    # fall-back). Fire when the arc arrives ON TIME: diff < 0 = early —
    # WAIT (t_star falls ~1:1 with wall time while t_pass barely moves, so
    # the difference closes on its own); diff > +tol = late no matter what
    # — a side-dodge if committed, else the anchor cycle. One-sided
    # acceptance here fired arcs up to 0.12s early = ~22px of extra fall
    # at the anchor.
    # Deep ring whose anchor is unreachable (below the survivable floor):
    # pre-position an over-dodge while there's still time. The trigger
    # doubles as a climb driver — from below it fires every echo-suppress
    # tick and each bob nets ~20px of height, so a climb started a couple
    # of bobs out clears the band top before the window.
    if anchor_y > floor_cap + 2.0:
        over_target = hole_top - DODGE_MARGIN_PX
        if over_target - rise >= CEILING_MARGIN_PX:
            return (not ascending and y_land_t >= over_target,
                    "dodge_over", None)

    if not ascending:
        apex_if_now = y_land_t - rise
        t_pass = (ACT_LATENCY_S + t_apex
                  + _fall_time(work_y - apex_if_now, g))
        if t_star <= t_apex + ACT_LATENCY_S + _fall_time(rise, g) + 0.3:
            diff = t_pass - t_star
            if abs(diff) <= SETUP_TOL_S:
                return (True, "setup", plan_frac)
            if diff < -SETUP_TOL_S:
                # about to breach survivable depth? flap rather than sink.
                if fly_y >= floor_cap:
                    return (True, "setup", plan_frac)
                return (False, "wait", plan_frac)
            # arriving late no matter what: this ring's launch won't align
            if committed:
                return _side_dodge()
            return (fly_y >= work_y, "hover", plan_frac)
        # Ring farther than one bob: natural anchor cycle (never committed
        # here — a far ring means a far window).
        return (fly_y >= work_y, "hover", plan_frac)

    return (False, "wait", plan_frac)


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
