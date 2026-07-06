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


class PlankLock:
    """Per-run lock on the plank-top row (#103).

    The plank overlay's y is FIXED for a run, but two adjacent strong tan
    rows (~185/190 in the 2026-07-06 captures) let the per-frame argmax
    flap between them — and at the false row the pit scan reads 50px pits
    as 7px fragments with wrong edges, which broke a jump (14:32 death),
    fed the steering slam a bogus target, and would corrupt the
    landing-solidity gate. Same pattern as the cart-x column prior: observe
    until stable, lock, re-detect under the lock constraint, release only
    on a sustained miss run (overlay gone = run over).

    update() is fed each frame's detected plank_y plus whether the grounded
    baseline is warmed — the cold-start frames before the cart is known
    read a stable FALSE band (y~296 on botrun_20260706_144046 f1-18), so
    stability alone must not lock.
    """

    def __init__(self, stable_n: int = 8, tol_px: int = 2,
                 max_misses: int = 20):
        self._recent: deque[int] = deque(maxlen=stable_n)
        self._tol = tol_px
        self._max_misses = max_misses
        self._lock: Optional[int] = None
        self._misses = 0

    @property
    def locked_y(self) -> Optional[int]:
        return self._lock

    def update(self, plank_y: Optional[int], baseline_ready: bool) -> None:
        if self._lock is None:
            if plank_y is None or not baseline_ready:
                self._recent.clear()
                return
            self._recent.append(int(plank_y))
            if (len(self._recent) == self._recent.maxlen and
                    max(self._recent) - min(self._recent) <= self._tol):
                self._lock = sorted(self._recent)[len(self._recent) // 2]
        elif plank_y is None:
            self._misses += 1
            if self._misses > self._max_misses:
                self._lock = None
                self._misses = 0
                self._recent.clear()
        else:
            self._misses = 0


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

# Plausibility band for a WINDOWED velocity estimate. Below 20 px/s the
# scene is frozen (death screen); above 250 px/s it's a detection jump, not
# scroll. Applied to the multi-frame window value only — NEVER to a single
# frame pair: obstacle x moves ~3px/frame with integer quantization, so
# individual pairs legitimately read 0 px, and rejecting those while keeping
# the fast pairs biased the estimate ~+30% high (the 2026-07-06 14:40 death:
# estimator said 105.7 px/s where wall-clock ground truth was ~80, inflating
# the pit window to [27,40] and firing at 39 — the cart landed 16px inside
# the far lip).
SCROLL_V_MIN, SCROLL_V_MAX = 20.0, 250.0
SCROLL_MAX_DT_S = 0.2       # frames farther apart than this break the chain
SCROLL_PLANK_JITTER_PX = 3  # plank_y moved more than this => mis-lock, break
SCROLL_WINDOW_FRAMES = 8    # (x, t) window the velocity is measured over
SCROLL_MIN_SPAN_S = 0.12    # window must span this long before it counts


class ScrollVelocity:
    """Live scroll-speed estimate (px/s) from the nearest-obstacle x track.

    Feed every frame's `find_next_terrain` result. Frames whose nearest
    obstacle is the SAME feature (same kind, non-rightward x step of
    plausible size, stable plank lock, small dt) extend a chain of (x, t)
    samples; velocity is the endpoint slope over the last
    SCROLL_WINDOW_FRAMES of the chain (~0.3s), EMA-smoothed. Measuring over
    the window instead of per frame pair lets the integer-quantized 0px
    steps average against the 6-7px ones — filtering per pair against a
    minimum speed silently drops only the slow samples and biases the
    estimate high (see the SCROLL_V_MIN comment). Identity switches (a pit
    scrolls past, the next obstacle pops in at a larger x), death-screen
    freezes, and plank mis-locks all break the chain or fail the windowed
    plausibility band, so no explicit obstacle tracker is needed.

    velocity() returns None until `warmup` window values have been
    accepted — callers fall back to the static (v_ref-tuned) windows.
    """

    def __init__(self, warmup: int = 8, alpha: float = 0.25):
        self._warmup = warmup
        self._alpha = alpha
        self._chain: deque[tuple] = deque(maxlen=SCROLL_WINDOW_FRAMES)  # (x, t)
        self._kind: Optional[str] = None
        self._plank: Optional[int] = None
        self._v: Optional[float] = None
        self._n = 0

    def update(self, terrain: Optional[dict], plank_y: Optional[int],
               now: float) -> None:
        if terrain is None or terrain.get("x") is None:
            self._chain.clear()  # a no-terrain frame breaks the chain
            self._kind = None
            self._plank = None
            return
        x, kind = terrain["x"], terrain.get("kind")
        prev_plank = self._plank
        self._plank = plank_y
        if self._chain:
            x_prev, t_prev = self._chain[-1]
            dt = now - t_prev
            dx = x - x_prev
            plank_moved = (prev_plank is None or plank_y is None or
                           abs(plank_y - prev_plank) > SCROLL_PLANK_JITTER_PX)
            if (kind != self._kind or plank_moved
                    or not (0.0 < dt <= SCROLL_MAX_DT_S)
                    or dx > 0                      # rightward: new obstacle
                    or dx < -(SCROLL_V_MAX * dt + 2)):  # too far: not scroll
                self._chain.clear()
        self._kind = kind
        self._chain.append((x, now))
        if len(self._chain) < 3:
            return
        x0, t0 = self._chain[0]
        x1, t1 = self._chain[-1]
        span = t1 - t0
        if span < SCROLL_MIN_SPAN_S:
            return
        v = (x0 - x1) / span  # leftward scroll => positive px/s
        if not (SCROLL_V_MIN <= v <= SCROLL_V_MAX):
            return  # frozen scene (all-zero window) or detection garbage
        self._v = v if self._v is None else (
            (1 - self._alpha) * self._v + self._alpha * v)
        self._n += 1

    def velocity(self) -> Optional[float]:
        return self._v if self._n >= self._warmup else None


# --- Landing-targeted pit fire ceiling (#105) ---
#
# Pit collision is CENTER-based (settled 2026-07-06; see mining_plan.md
# "Collision model"), and the click->touchdown time is ~constant:
# T ≈ 1.16-1.23s across every exactly-measurable arc (freeze-exact death
# clearances + wall-clock [traj] velocities; runs 17/22). So the landing
# CENTER's clearance past a jumped pit's far edge is predictable at fire
# time:  clear = v*T - dist - HALF_CART - W.
# Firing when dist first drops under the affine ceiling
#   D* = v*T - HALF_CART - W - CLEAR_TARGET
# puts the landing ~CLEAR_TARGET px past the far edge — instead of
# wherever the proportional window edge lands it (run 22's jump #4 fired
# at the v-scaled top and landed at clear = -1: dead by one pixel). The
# model postdicts all measured clearances within ~±5px, so CLEAR_TARGET
# = 12 keeps both sides safe: alive-margin 12±5 > 0, and the next
# gauntlet pit is still rescue-jumpable (gap 39 - clear 12 = 27 ≥ the
# ~20px center-based launch need).
ARC_T_TOTAL_S = 1.2          # click -> touchdown, speed-invariant
LAND_CLEAR_TARGET_PX = 12    # aim the landing center this far past the far edge
CART_HALF_WIDTH_PX = 18
# Sanity clamp on the affine ceiling: below 20 the ceiling is meaningless
# (fire ASAP); above 55 a v-estimate spike is inflating it (never observed
# legitimately — at v=110, W=50 the ceiling is 52).
PIT_CEILING_CLAMP = (20, 55)


def pit_fire_ceiling(v: Optional[float], width: Optional[int]) -> Optional[int]:
    """The landing-targeted fire ceiling D* for a pit of the given width at
    live speed v — replaces the proportional window top for pits. None when
    v (estimator warming up) or width is unavailable; callers fall back to
    the v-scaled top."""
    if v is None or width is None:
        return None
    d = v * ARC_T_TOTAL_S - CART_HALF_WIDTH_PX - width - LAND_CLEAR_TARGET_PX
    lo, hi = PIT_CEILING_CLAMP
    return int(round(min(max(d, lo), hi)))


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
                ore_fallback_min: int = 0,
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
    (see should_slam) — and ALSO in the pit window as a survival fallback: an
    ore can emerge below its own window when the cart was airborne (a
    slam-rebound arc) through the whole [ore_trig_min, ore_trig_max] approach,
    and a grounded cart that just waits eats the ore and dies (the 15:00
    2026-07-06 death: landed from a rebound with the next ore at 29px, ore
    window [46,80] — no branch fired). The pit-style late jump flies over it;
    if the arc lines up, should_slam converts the flyover into a scoring slam
    for free. With the ore window omitted (None) ore never jumps, so the
    pit-only behaviour is unchanged."""
    if cart is None or plank_y is None or terrain is None:
        return False
    kind = terrain.get("kind")
    dist = terrain.get("distance_px")
    if dist is None:
        return False
    if kind == "pit":
        in_window = trig_min <= dist <= trig_max
    elif kind == "ore" and ore_trig_min is not None and ore_trig_max is not None:
        # The fallback's low edge is clamped to ore_fallback_min: pits
        # forgive a low-floor rescue (center-based collision) but ore kills
        # on FRONT contact, so the cart must be airborne before the ore
        # reaches its bumper — the edge-based delta*v floor.
        fb_lo = max(trig_min, ore_fallback_min)
        in_window = (ore_trig_min <= dist <= ore_trig_max or
                     fb_lo <= dist <= trig_max)
    else:
        return False
    if not in_window:
        return False
    if now - last_click_time < cooldown_s:
        return False
    if is_cart_airborne(cart[1], grounded_baseline_y, airborne_eps):
        return False
    return True


# Steering slam (#104): while airborne with a PIT ahead in this distance
# band (px at SCROLL_V_REF_PIT, scaled live like the other windows), slam
# onto the bare plank to land short of the pit and re-arm the grounded
# jump. Slamming onto empty plank is a safe landing (user-verified
# 2026-07-06). The band's edges come from the slam descent costing the pit
# ~0.4s * v ~= 38px of approach:
#   - below MIN the slam would land at/inside the pit — but so does riding
#     the arc, so firing there only mislabels doomed rows; skip.
#   - above MAX the natural landing leaves the pit >= the jump window
#     anyway — riding is safe and preserves a scoring rebound.
# Motivating death (run 19, botrun_20260706_150731): rebound descended
# into a pit that approached 63->15 during the arc; touchdown at the lip
# left no legal move (launch needs ~19px lead at v~96). A steer slam at
# first sight (63px) lands with the pit at ~25 — inside the jump window.
#
# MIN lowered 50 -> 40 after run 20 (botrun_20260706_151934): in a two-pit
# spacing trap the second pit's FIRST sighting was 54px (scaled band bottom
# was 63 — no fire) and the natural landing died at 10px. The 0.4s descent
# cost that justified 50 was measured from rebound apex (~90px); from
# jump-arc descent heights (~70px) the slam costs only ~25-30px of
# approach, so a steer at ~50 lands at ~20-25 — tight but jumpable
# (JUMP_TRIGGER_MIN sits at the physics floor). Below ~40 both slam and
# ride land in the pit; keep not firing there.
STEER_SLAM_MIN = 40
STEER_SLAM_MAX = 75


def should_steer_slam(*, cart, terrain, now: float, last_slam_time: float,
                      grounded_baseline_y: Optional[int],
                      slam_cooldown_s: float,
                      steer_min: int, steer_max: int,
                      landing_solid: Optional[bool],
                      airborne_eps: int = JUMP_GROUNDED_EPS_PX) -> bool:
    """The steering slam (#104) — cut an arc short over bare plank so an
    incoming pit is met grounded, with jump lead in hand. Returns True iff
    the cart is AIRBORNE, the nearest obstacle is a PIT inside the steer
    band, the ground DIRECTLY BELOW the cart is solid plank, and the
    per-arc slam cooldown has expired.

    landing_solid is the under-cart check (detector.plank_dark_fraction
    over the cart's x-span): the ahead-only terrain scan cannot see a pit
    already beneath the airborne cart, and slamming onto one is the exact
    death this predicate exists to prevent — run 21
    (botrun_20260706_152412) steer-slammed into the first pit of a
    three-pit gauntlet that the ahead-scan reported as a pit 70px out.
    Anything but an affirmative True (None = row unlocked / band unusable)
    rides the arc instead: the natural landing is never worse than a slam
    onto the same spot sooner.

    Distinct from should_slam (the scoring slam): that one requires ORE
    under the cart and must never fire over a pit; this one requires the
    pit to still be well AHEAD (>= steer_min) so the slam lands on wood
    before it. Callers check should_slam first — a slammable ore both
    scores and gets the cart down. DB rows distinguish the two by
    next_kind (ore = scoring slam, pit = steer slam)."""
    if cart is None or terrain is None:
        return False
    if terrain.get("kind") != "pit":
        return False
    if landing_solid is not True:
        return False
    if not is_cart_airborne(cart[1], grounded_baseline_y, airborne_eps):
        return False
    dist = terrain.get("distance_px")
    if dist is None or not (steer_min <= dist <= steer_max):
        return False
    if now - last_slam_time < slam_cooldown_s:
        return False
    return True


# A scoring slam must fall onto the ore's TOP, which takes real altitude:
# every successful slam (6/6, runs 16-23) fired from 69-76px above rest,
# while the one slam fired from ~22px (run 23's SLAM #7 — the first frame
# past the 18px airborne threshold, ore still at the front bumper behind
# the 10px dist floor) dropped the cart IN FRONT of the ore and died to
# the side hit. Requiring this much height means a low flyover (the ore
# pit-window fallback jump) just flies over: by the time the arc passes
# 40px the ore has scrolled under or behind, so the slam fires in proper
# geometry or not at all. User-observed as "the last jump slam was a
# little too quick".
SLAM_MIN_HEIGHT_PX = 40


def should_slam(*, cart, terrain, now: float, last_slam_time: float,
                grounded_baseline_y: Optional[int], slam_cooldown_s: float,
                slam_max_dist: int,
                airborne_eps: int = JUMP_GROUNDED_EPS_PX,
                min_height: int = SLAM_MIN_HEIGHT_PX) -> bool:
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
    # Enough altitude to come down ON the ore, not beside it (see
    # SLAM_MIN_HEIGHT_PX). grounded_baseline_y is not None here — the
    # airborne check above already returned False otherwise.
    if grounded_baseline_y - cart[1] < min_height:
        return False
    dist = terrain.get("distance_px")
    if dist is None or dist > slam_max_dist:
        return False
    if now - last_slam_time < slam_cooldown_s:
        return False
    return True
