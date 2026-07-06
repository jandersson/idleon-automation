"""Pure jump-policy logic for the mining bot — no IO, no CV.

Split out from main.py so the fire decision is unit-testable in isolation
(the loop just samples the detector and asks should_jump). The load-bearing
piece is the airborne guard: in the cart minigame a click while the cart is
GROUNDED is a jump, but a click while it's AIRBORNE is a *slam* (a fast
descent). The cooldown (JUMP_COOLDOWN_S) only blanks the first ~0.6s after a
jump, but the arc stays airborne ~0.9s — so a second pit entering the
trigger window in that ~0.3s tail used to pass the predicate and fire a
click the game read as a slam, driving the cart down into the approaching
pit. That is the mechanism behind the documented "spaced-obstacle" death
(see docs/mining_plan.md). should_jump refuses to fire while airborne.

Altitude is read from cart_y against a grounded BASELINE rather than from
(plank_y - cart_y): plank_y is rock-stable while grounded but jitters a few
px during the jump (exactly when the difference would be near the launch/
land boundary), whereas the grounded cart_y is extremely stable (193 for
every grounded frame of botrun_20260615_012340) and the peak is far away
(~133), so a baseline-relative test has a wide, noise-proof margin.
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Tuple


# A cart this many px ABOVE its grounded resting y counts as airborne.
# Screen y grows downward, so a raised (airborne) cart has a SMALLER y. The
# grounded->peak swing is ~60px (193->133) so a generous 18px threshold
# cleanly brackets the resting jitter while still catching the whole up-arc.
JUMP_GROUNDED_EPS_PX = 18

# How many consecutive below-level window maxes the grounded level holds
# for before falling back to the plain window max. The hold exists for
# slam-rebound chains (longest observed: ~66 frames / 2.7s airborne without
# grounding, botrun_20260616_135951); the cap exists because the hold can
# also be entered via a one-off inflated sample (a false cart match below
# the plank) and an uncapped hold would then pin an inflated level — and
# an inflated level makes the TRUE grounded y read as airborne, suppressing
# every jump for the rest of the run. 200 samples (~5-8s at live FPS)
# comfortably covers real chains while bounding the pathological case.
GROUNDED_FREEZE_MAX_SAMPLES = 200


class GroundedBaseline:
    """Rolling estimate of the cart's resting (grounded) screen-y.

    The cart sits at its MAX y when grounded and rises (smaller y) on a jump;
    a death pushes it slightly below the plank (a bit larger y) just before
    it vanishes. So the grounded level is the high-y end of a recent window,
    and the MAX over the window tracks it robustly: airborne frames (small y)
    can only ever pull a mean/median down (risking a false 'airborne' verdict
    on a grounded cart — the dangerous direction), but they never raise the
    max. The occasional death frame inflates the max a few px, which only
    makes the guard *less* eager to suppress — the safe direction.

    The established level NEVER moves down with the window max: a
    slam-rebound chain keeps the cart off the ground longer than the window
    (66 frames on botrun_20260616_135951 — jump -> slam -> rebound -> slam),
    so a plain window-max decays to an airborne y. That corrupts more than
    the guard: the baseline anchors the plank search
    (`_find_plank_top_y(near_y=...)`), and the decayed value let a false tan
    band at y=149 outscore the true plank at 190 (replay frames 141-146),
    freezing terrain output at a spurious slam-range distance. An
    eps-tolerance downward rule is NOT enough — slam contacts bounce at
    ore-top height (~20px above grounded, y~155-161 vs 179), and each one
    steps the window max down within any reasonable eps, ratcheting the
    level to an airborne y anyway (observed on the same replay). So
    downward: hold, and only fall back to the window max after
    GROUNDED_FREEZE_MAX_SAMPLES consecutive lower-max updates (recovery
    path for a one-off inflated sample). Upward (cart sank / death frame):
    adopt immediately — the safe direction, as before.

    baseline() returns None until `warmup` samples have been seen, so the
    airborne guard defaults to 'grounded' (never suppresses a survival jump)
    during the first frames of a run before the resting level is known.
    """

    def __init__(self, window: int = 45, warmup: int = 8):
        self._buf: deque[int] = deque(maxlen=window)
        self._warmup = warmup
        self._level: Optional[int] = None
        self._frozen_for = 0

    def update(self, cart_y: Optional[int]) -> None:
        """Feed this frame's cart_y (None when the cart isn't detected —
        ignored, so a lost cart doesn't corrupt the resting estimate)."""
        if cart_y is None:
            return
        self._buf.append(int(cart_y))
        if len(self._buf) < self._warmup:
            return
        m = max(self._buf)
        if self._level is None or m >= self._level:
            self._level = m
            self._frozen_for = 0
        else:
            # The window max sits below the established level — grounded
            # frames aged out mid-chain (or the level was inflated by a
            # one-off bad sample). Hold the level; it re-anchors on the next
            # at-level sample, or falls back to the window max after the
            # safety cap (see GROUNDED_FREEZE_MAX_SAMPLES for why the cap
            # must exist). No eps tolerance here: slam-contact bounces step
            # the max down ~10-20px at a time and ratchet any eps rule.
            self._frozen_for += 1
            if self._frozen_for > GROUNDED_FREEZE_MAX_SAMPLES:
                self._level = m
                self._frozen_for = 0

    def baseline(self) -> Optional[int]:
        return self._level


def is_cart_airborne(cart_y: Optional[int], grounded_baseline_y: Optional[int],
                     eps: int = JUMP_GROUNDED_EPS_PX) -> bool:
    """True when the cart is clearly above its grounded resting level.

    Unknown inputs (no cart, no warmed-up baseline) return False — i.e. treat
    as grounded — so the airborne guard never suppresses a needed survival
    jump just because altitude is momentarily unknown. The conservative
    default is 'fire if in doubt'."""
    if cart_y is None or grounded_baseline_y is None:
        return False
    return cart_y < grounded_baseline_y - eps


# --- Scroll-velocity estimation + speed-adaptive trigger scaling (#53) ---
#
# The track's scroll speed RAMPS within a run: replaying the scoring run
# botrun_20260616_135951 at its live ~24.5 FPS cadence gives ~82 px/s on the
# first-pit approach (frames 19-44), ~86-89 mid-run, ~117 px/s by frames
# 147-161 — a ~40% ramp inside six seconds. The trigger constants are
# DISTANCES tuned at a specific speed, but the underlying physics bounds are
# TIMES (click->launch delta ~0.23s, airborne span ~0.88s — cart physics,
# speed-invariant), so every distance window scales linearly with v: at
# v=100 the pit bound D >= delta*v needs >=23 px and the static [20,30]
# window's low edge is already lethal. Scaling windows by v / v_ref keeps
# the TIME behaviour the constants encoded at the speed they were tuned at.
#
# Velocity is measured in px/SECOND against wall-clock time, not px/frame —
# per-frame units would silently rescale with every loop-speed change (the
# darts px/poll lesson, CLAUDE.md "Units").

# Speeds the static distance constants were tuned at. The pit window [20,30]
# comes from the Run-11 absolute-time derivation at v~79 (trace
# session_20260616_131320); the ore window / slam distance come from the
# scoring run's human timings at v~88 (mid-run average of the ramp above).
SCROLL_V_REF_PIT = 79.0
SCROLL_V_REF_ORE = 88.0

# Plausibility band for a single-frame velocity sample. Below 20 px/s the
# scene is frozen (death screen, plank mis-lock) or the sample straddles an
# obstacle-identity switch; above 250 px/s it's a detection jump, not scroll.
SCROLL_V_MIN, SCROLL_V_MAX = 20.0, 250.0
SCROLL_MAX_DT_S = 0.2       # samples farther apart than this don't pair
SCROLL_PLANK_JITTER_PX = 3  # plank_y moved more than this => mis-lock, reject


class ScrollVelocity:
    """Live scroll-speed estimate (px/s) from the nearest-obstacle x track.

    Feed every frame's `find_next_terrain` result; consecutive frames whose
    nearest obstacle is the SAME feature (same kind, leftward x delta inside
    the plausibility band, stable plank lock) contribute (x0-x1)/dt samples
    to an EMA. Identity switches (a pit scrolls past, the next obstacle pops
    in at a larger x), frozen death screens, and plank mis-locks are all
    rejected by the gates, so no explicit obstacle tracker is needed.

    velocity() returns None until `warmup` samples have been accepted —
    callers fall back to the static (v_ref-tuned) windows until then.
    """

    def __init__(self, warmup: int = 8, alpha: float = 0.25):
        self._warmup = warmup
        self._alpha = alpha
        self._prev: Optional[tuple] = None  # (x, kind, plank_y, t)
        self._v: Optional[float] = None
        self._n = 0

    def update(self, terrain: Optional[dict], plank_y: Optional[int],
               now: float) -> None:
        prev = self._prev
        cur = None
        if terrain is not None and terrain.get("x") is not None:
            cur = (terrain["x"], terrain.get("kind"), plank_y, now)
        self._prev = cur  # a no-terrain frame breaks the pairing chain
        if prev is None or cur is None:
            return
        x0, k0, p0, t0 = prev
        x1, k1, p1, t1 = cur
        dt = t1 - t0
        if not (0.0 < dt <= SCROLL_MAX_DT_S):
            return
        if k1 != k0:
            return
        if p0 is None or p1 is None or abs(p1 - p0) > SCROLL_PLANK_JITTER_PX:
            return
        v = (x0 - x1) / dt  # leftward scroll => positive px/s
        if not (SCROLL_V_MIN <= v <= SCROLL_V_MAX):
            return
        self._v = v if self._v is None else (
            (1 - self._alpha) * self._v + self._alpha * v)
        self._n += 1

    def velocity(self) -> Optional[float]:
        return self._v if self._n >= self._warmup else None


def scale_window(lo: int, hi: int, v: Optional[float],
                 v_ref: float) -> Tuple[int, int]:
    """Scale a [lo, hi] trigger-distance window tuned at v_ref to the live
    scroll speed v. v=None (estimator not warmed up) returns the static
    window unchanged — identical behaviour to the pre-adaptive bot."""
    if v is None:
        return lo, hi
    r = v / v_ref
    return int(round(lo * r)), int(round(hi * r))


def scale_slam_dist(max_dist: int, v: Optional[float], v_ref: float) -> int:
    """Scale the slam fire distance, but never below the static value:
    find_next_terrain floors ore distance at SCAN_BUFFER_PX(=10), so a
    scaled-down window below ~11 could never fire at all."""
    if v is None:
        return max_dist
    return max(max_dist, int(round(max_dist * v / v_ref)))


def should_jump(*, cart, plank_y, terrain, now: float, last_click_time: float,
                grounded_baseline_y: Optional[int], cooldown_s: float,
                trig_min: int, trig_max: int,
                ore_trig_min: Optional[int] = None,
                ore_trig_max: Optional[int] = None,
                airborne_eps: int = JUMP_GROUNDED_EPS_PX) -> bool:
    """The grounded-jump predicate (the FIRST click). Returns True iff a jump
    should fire this frame.

    Gates, cheap-to-expensive: cart + plank present; the nearest terrain is a
    jumpable obstacle in its trigger window; the post-jump cooldown has
    expired; the cart is NOT airborne (an airborne click is a slam, see
    should_slam + the module docstring). Click POSITION is irrelevant (Idleon
    reads a click as a button press), so this returns only yes/no.

    A PIT fires in [trig_min, trig_max] (jump to clear it). An ORE fires in
    [ore_trig_min, ore_trig_max] when those are supplied — a FARTHER window so
    the cart is airborne by the time the ore scrolls under it to be slammed
    (see should_slam). With the ore window omitted (None) ore never jumps, so
    the pit-only behaviour is unchanged."""
    if cart is None or plank_y is None or terrain is None:
        return False
    kind = terrain.get("kind")
    dist = terrain.get("distance_px")
    if dist is None:
        return False
    if kind == "pit":
        lo, hi = trig_min, trig_max
    elif kind == "ore" and ore_trig_min is not None and ore_trig_max is not None:
        lo, hi = ore_trig_min, ore_trig_max
    else:
        return False
    if not (lo <= dist <= hi):
        return False
    if now - last_click_time < cooldown_s:
        return False
    if is_cart_airborne(cart[1], grounded_baseline_y, airborne_eps):
        return False
    return True


def should_slam(*, cart, terrain, now: float, last_slam_time: float,
                grounded_baseline_y: Optional[int], slam_cooldown_s: float,
                slam_max_dist: int,
                airborne_eps: int = JUMP_GROUNDED_EPS_PX) -> bool:
    """The slam predicate (the SECOND click) — drop the airborne cart onto
    ore to mine it. Returns True iff:

      - the cart is AIRBORNE (slamming is only meaningful mid-arc), and
      - the nearest obstacle is ORE within slam_max_dist of the cart's
        leading edge (the ore has scrolled under the cart), and
      - the per-arc slam cooldown has expired.

    NEVER fires over a pit (kind must be 'ore') — an airborne click over a
    pit is the lethal slam the airborne guard exists to prevent. The grounded
    jump that got the cart airborne is should_jump's ore branch; this is the
    follow-up. Self-timing: it fires when the ore actually reaches the cart,
    not on a fixed delay."""
    if cart is None or terrain is None:
        return False
    if terrain.get("kind") != "ore":
        return False
    if not is_cart_airborne(cart[1], grounded_baseline_y, airborne_eps):
        return False
    dist = terrain.get("distance_px")
    if dist is None or dist > slam_max_dist:
        return False
    if now - last_slam_time < slam_cooldown_s:
        return False
    return True
