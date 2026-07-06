"""Tests for minigames/mining/policy.py — the pure jump-fire predicate and
the grounded-baseline / airborne guard."""
import glob
from pathlib import Path

import cv2

from minigames.mining.policy import (
    GROUNDED_FREEZE_MAX_SAMPLES,
    GroundedBaseline,
    JUMP_GROUNDED_EPS_PX,
    SCROLL_V_REF_ORE,
    SCROLL_V_REF_PIT,
    ScrollVelocity,
    is_cart_airborne,
    scale_slam_dist,
    scale_window,
    should_jump,
    should_slam,
    should_steer_slam,
)
from minigames.mining import detector as D

PIT = {"kind": "pit", "x": 300, "distance_px": 40}
BASE_KW = dict(
    cart=(150, 193), plank_y=200, terrain=PIT, now=10.0, last_click_time=0.0,
    grounded_baseline_y=193, cooldown_s=0.6, trig_min=30, trig_max=50,
)


def test_fires_on_grounded_pit_in_window_off_cooldown():
    assert should_jump(**BASE_KW) is True


def test_no_fire_when_no_cart_or_plank_or_terrain():
    assert should_jump(**{**BASE_KW, "cart": None}) is False
    assert should_jump(**{**BASE_KW, "plank_y": None}) is False
    assert should_jump(**{**BASE_KW, "terrain": None}) is False


def test_no_fire_on_ore_or_out_of_window():
    assert should_jump(**{**BASE_KW, "terrain": {"kind": "ore", "x": 300, "distance_px": 40}}) is False
    assert should_jump(**{**BASE_KW, "terrain": {"kind": "pit", "x": 300, "distance_px": 80}}) is False
    assert should_jump(**{**BASE_KW, "terrain": {"kind": "pit", "x": 300, "distance_px": 10}}) is False


def test_trigger_window_fires_later_than_the_old_far_lip():
    """#5 (2026-06-16): the configured trigger fires LATER (smaller pit_dist)
    than the old [30,50], so the airborne window is phased to contain the gap
    passage instead of landing into the far edge. The old fire distance (~49)
    must no longer fire; a pit at the new max does. MIN stays >= ~18 (the
    delta*v floor) so the cart is airborne before the gap arrives."""
    from minigames.mining.main import JUMP_TRIGGER_MIN, JUMP_TRIGGER_MAX
    assert JUMP_TRIGGER_MAX <= 32
    assert JUMP_TRIGGER_MIN >= 18
    kw = {**BASE_KW, "trig_min": JUMP_TRIGGER_MIN, "trig_max": JUMP_TRIGGER_MAX}
    far = {**kw, "terrain": {"kind": "pit", "x": 300, "distance_px": 49}}
    assert should_jump(**far) is False  # the old fire point no longer fires
    at_max = {**kw, "terrain": {"kind": "pit", "x": 300, "distance_px": JUMP_TRIGGER_MAX}}
    assert should_jump(**at_max) is True


def test_jump_fires_for_ore_only_in_the_ore_window():
    """#5 ore: a grounded jump fires for ore in [ore_trig_min, ore_trig_max]
    (a farther window than pits — be airborne when it arrives). Without the
    ore window params, ore never jumps (pit-only behaviour preserved)."""
    ore = {"kind": "ore", "x": 300, "distance_px": 60}  # in [40,70], not pit [30,50]
    kw = {**BASE_KW, "terrain": ore, "ore_trig_min": 40, "ore_trig_max": 70}
    assert should_jump(**kw) is True
    # ore below BOTH windows doesn't fire (too close to launch in time)
    assert should_jump(**{**kw, "terrain": {"kind": "ore", "x": 300, "distance_px": 20}}) is False
    # backward-compat: no ore window -> ore never jumps
    assert should_jump(**{**BASE_KW, "terrain": ore}) is False


def test_too_close_ore_fires_the_pit_window_fallback():
    """Regression for the 15:00 2026-07-06 death: the slam-rebound arc kept
    the cart airborne through the ore's whole [46,80] window; it landed with
    the ore at ~29px and no branch fired — the ore crashed the grounded cart
    (frames botrun_20260706_150022, freeze at f102 with the ore mashed into
    the cart's front). An ore in the PIT window must fire a survival jump."""
    kw = {**BASE_KW, "trig_min": 25, "trig_max": 38,
          "ore_trig_min": 46, "ore_trig_max": 80}
    ore_close = {"kind": "ore", "x": 300, "distance_px": 29}
    assert should_jump(**{**kw, "terrain": ore_close}) is True
    # Between the windows: no fire yet (it enters the pit window frames later).
    ore_gap = {"kind": "ore", "x": 300, "distance_px": 42}
    assert should_jump(**{**kw, "terrain": ore_gap}) is False
    # The fallback still respects the airborne guard.
    assert should_jump(**{**kw, "terrain": ore_close, "cart": (150, 133)}) is False


def test_should_slam_only_airborne_over_near_ore():
    """The slam (2nd click) fires only when airborne AND the nearest obstacle
    is ore within slam_max_dist — never over a pit (lethal), never grounded."""
    base = dict(cart=(150, 120), terrain={"kind": "ore", "x": 160, "distance_px": 15},
                now=10.0, last_slam_time=0.0, grounded_baseline_y=193,
                slam_cooldown_s=0.5, slam_max_dist=25)
    assert should_slam(**base) is True                                   # airborne + near ore
    assert should_slam(**{**base, "cart": (150, 193)}) is False           # grounded -> no slam
    assert should_slam(**{**base, "terrain": {"kind": "pit", "x": 160, "distance_px": 15}}) is False  # never over a pit
    assert should_slam(**{**base, "terrain": {"kind": "ore", "x": 200, "distance_px": 40}}) is False  # ore too far
    assert should_slam(**{**base, "now": 10.2, "last_slam_time": 10.0}) is False  # within slam cooldown


def test_steer_slam_fires_airborne_on_incoming_pit_in_band():
    """#104: airborne with a pit ahead inside the steer band -> slam onto
    bare plank (safe, user-verified 2026-07-06) to land short and re-arm the
    grounded jump. Motivating death: run 19's rebound descended into a pit
    first seen at 63px; the steer band must catch it."""
    base = dict(cart=(150, 120), now=10.0, last_slam_time=0.0,
                grounded_baseline_y=193, slam_cooldown_s=0.5,
                steer_min=49, steer_max=91)  # [40,75] scaled to v~96
    pit_at_63 = {"kind": "pit", "x": 300, "distance_px": 63}
    assert should_steer_slam(**base, terrain=pit_at_63) is True
    # Run 20's two-pit trap: second pit first sighted at 54 during the jump
    # arc — inside the lowered band (the old [50,75] bottom missed it).
    assert should_steer_slam(**base, terrain={"kind": "pit", "x": 300, "distance_px": 54}) is True
    # Grounded -> never (this is an airborne-only rescue).
    assert should_steer_slam(**{**base, "cart": (150, 193)}, terrain=pit_at_63) is False
    # Below the band the slam would land at/inside the pit — don't fire.
    assert should_steer_slam(**base, terrain={"kind": "pit", "x": 300, "distance_px": 40}) is False
    # Above the band the natural landing is safe — preserve the rebound.
    assert should_steer_slam(**base, terrain={"kind": "pit", "x": 300, "distance_px": 100}) is False
    # Ore is the scoring slam's business, not steering's.
    assert should_steer_slam(**base, terrain={"kind": "ore", "x": 300, "distance_px": 63}) is False
    # Per-arc slam cooldown still applies.
    assert should_steer_slam(**{**base, "now": 10.0, "last_slam_time": 9.8},
                             terrain=pit_at_63) is False


def test_no_fire_during_cooldown():
    assert should_jump(**{**BASE_KW, "now": 10.0, "last_click_time": 9.8}) is False  # 0.2s < 0.6s
    assert should_jump(**{**BASE_KW, "now": 10.0, "last_click_time": 9.3}) is True   # 0.7s >= 0.6s


def test_airborne_guard_suppresses_mid_arc_slam():
    """The load-bearing fix: a pit in the trigger window while the cart is
    airborne must NOT fire (an airborne click is a lethal slam)."""
    airborne = {**BASE_KW, "cart": (150, 133)}  # peak, 60px above grounded 193
    assert should_jump(**airborne) is False
    # A barely-airborne cart (just over eps) is still suppressed.
    just_up = {**BASE_KW, "cart": (150, 193 - JUMP_GROUNDED_EPS_PX - 1)}
    assert should_jump(**just_up) is False


def test_airborne_guard_defaults_to_grounded_when_baseline_unknown():
    """No warmed-up baseline -> treat as grounded so a survival jump is never
    suppressed for lack of altitude info."""
    assert is_cart_airborne(133, None) is False
    assert should_jump(**{**BASE_KW, "grounded_baseline_y": None}) is True


def test_is_cart_airborne_threshold():
    assert is_cart_airborne(193, 193) is False           # at rest
    assert is_cart_airborne(193 - JUMP_GROUNDED_EPS_PX, 193) is False  # exactly eps, not yet over
    assert is_cart_airborne(193 - JUMP_GROUNDED_EPS_PX - 1, 193) is True
    assert is_cart_airborne(None, 193) is False           # unknown cart -> grounded


def test_grounded_baseline_warmup_and_max():
    gb = GroundedBaseline(window=45, warmup=8)
    assert gb.baseline() is None
    for _ in range(7):
        gb.update(193)
    assert gb.baseline() is None        # still warming up
    gb.update(193)
    assert gb.baseline() == 193         # warmed up
    # An airborne dip (small y) must NOT lower the baseline (the dangerous
    # direction — it would falsely flag a grounded cart as airborne).
    gb.update(133)
    assert gb.baseline() == 193
    # None (cart lost) is ignored.
    gb.update(None)
    assert gb.baseline() == 193


def test_grounded_baseline_ignores_airborne_majority_within_window():
    """Even if recent frames are mostly airborne, max keeps the resting y so
    the guard can't falsely suppress once the cart lands."""
    gb = GroundedBaseline(window=45, warmup=8)
    for _ in range(8):
        gb.update(193)          # establish grounded
    for _ in range(20):
        gb.update(135)          # a long airborne stretch
    assert gb.baseline() == 193  # resting level preserved


def test_grounded_baseline_holds_through_a_rebound_chain():
    """A slam-rebound chain keeps the cart airborne longer than the window
    (66 frames on botrun_20260616_135951) — the level must NOT decay to an
    airborne y (that mis-anchored the plank search to a false band at y=149
    and froze terrain output at slam range)."""
    gb = GroundedBaseline(window=45, warmup=8)
    for _ in range(8):
        gb.update(179)
    for _ in range(70):          # airborne chain longer than the 45 window
        gb.update(110)
    assert gb.baseline() == 179  # held at the true grounded level
    gb.update(180)               # landing (1px sink) re-anchors upward
    assert gb.baseline() == 180


def test_grounded_baseline_resists_the_slam_contact_ratchet():
    """Slam contacts bounce the cart at ore-top height (y~155-161, ~20px
    above grounded 179) — a staircase of window maxes that stepped an
    eps-tolerant rule down to an airborne level on the replay. The level
    must hold at 179 through the measured bounce pattern."""
    gb = GroundedBaseline(window=45, warmup=8)
    for _ in range(20):
        gb.update(179)           # grounded run-up
    # jump arc -> slam contact bounce -> rebound arc -> second contact,
    # shaped like botrun_20260616_135951 frames 96-162
    chain = ([161, 159, 146, 143, 141, 129, 128, 121, 119, 119, 115, 114] +
             [114, 116, 122, 125, 135, 152, 155, 147, 147] +      # slam 1 down
             [135, 125, 120, 119, 110, 107, 104, 100, 100, 98] +  # rebound up
             [98, 99, 102, 103, 109, 111, 123, 140, 144, 158] +   # slam 2 down
             [153, 151, 133, 133, 123, 120, 111, 110, 104, 99] +  # rebound up
             [98, 97, 97, 98, 100, 102, 108, 114, 118, 125])
    for y in chain:
        gb.update(y)
    assert gb.baseline() == 179  # no ratchet-down to a bounce height


def test_grounded_baseline_freeze_cap_recovers_inflated_level():
    """The freeze is capped: a one-off inflated sample (false cart match
    below the plank) must not pin an inflated level forever — an inflated
    level reads the TRUE grounded y as airborne and would suppress every
    jump. After the cap, the level falls back to the window max."""
    gb = GroundedBaseline(window=45, warmup=8)
    for _ in range(8):
        gb.update(193)
    gb.update(240)               # spike far below the plank (inflated max)
    assert gb.baseline() == 240
    for _ in range(45 + GROUNDED_FREEZE_MAX_SAMPLES + 2):
        gb.update(193)           # cart is really grounded the whole time
    assert gb.baseline() == 193  # spike aged out + freeze cap expired


# --- ScrollVelocity + speed-adaptive window scaling (#53) ---

FPS = 24.5  # live loop cadence of the fixture run
# Nearest-obstacle x per frame, replayed from botrun_20260616_135951 through
# the live detector (real data, no frame archaeology needed at test time).
PIT_TRACK_EARLY = [436, 433, 428, 426, 423, 419, 412, 409, 407, 402, 400,
                   398, 393, 392, 388, 384, 381, 377, 376, 371, 368, 365,
                   361, 357, 355, 352]          # frames 19-44, ~82 px/s
ORE_TRACK_LATE = [419, 412, 405, 399, 396, 393, 389, 384, 381, 373, 368,
                  365, 360, 358, 352]           # frames 147-161, ~117 px/s


def _feed(sv, xs, kind="pit", plank_y=190, fps=FPS, t0=0.0):
    for i, x in enumerate(xs):
        sv.update({"kind": kind, "x": x, "distance_px": 0}, plank_y, t0 + i / fps)


def test_scroll_velocity_measures_the_replayed_pit_approach():
    sv = ScrollVelocity()
    _feed(sv, PIT_TRACK_EARLY)
    v = sv.velocity()
    assert v is not None
    assert 65 <= v <= 100        # segment ground truth ~82 px/s


def test_scroll_velocity_tracks_the_late_run_ramp():
    """Fed the late-run ore track, the estimate must land clearly above the
    early-run speed — the within-run ramp (~82 -> ~117 px/s) is the whole
    reason the windows scale."""
    sv = ScrollVelocity()
    _feed(sv, ORE_TRACK_LATE, kind="ore")
    v = sv.velocity()
    assert v is not None
    assert v >= 95


def test_scroll_velocity_warmup_returns_none():
    sv = ScrollVelocity(warmup=8)
    _feed(sv, PIT_TRACK_EARLY[:6])   # only 5 accepted deltas < warmup
    assert sv.velocity() is None


def test_scroll_velocity_rejects_identity_switches_and_mislocks():
    """A kind change, a rightward x jump (next obstacle pops in), a frozen
    scene (death screen), and a plank mis-lock frame must all contribute
    nothing."""
    sv = ScrollVelocity(warmup=1)
    _feed(sv, PIT_TRACK_EARLY[:8], kind="pit")
    v0 = sv.velocity()
    assert v0 is not None
    sv.update({"kind": "ore", "x": 500, "distance_px": 0}, 190, 100.0)
    sv.update({"kind": "ore", "x": 620, "distance_px": 0}, 190, 100.04)  # rightward
    # A frozen scene: the chain builds but the windowed velocity is 0,
    # below the plausibility floor.
    for j in range(10):
        sv.update({"kind": "ore", "x": 620, "distance_px": 0}, 190, 100.08 + j / FPS)
    sv.update({"kind": "ore", "x": 616, "distance_px": 0}, 149, 101.0)  # plank jumped
    assert sv.velocity() == v0   # none of those moved the estimate


# Nearest-pit x per frame from botrun_20260706_144046 (the 14:40 death run),
# frames 2-32 (grounded approach up to the fire). Wall-clock ground truth for
# this stretch is ~80 px/s (telemetry: 39px over 0.485s pre-fire; traj: 16px
# over 0.20s post-fire). The track has four 0px frame steps (integer
# quantization at ~3px/frame) mixed with 6-7px ones.
PIT_TRACK_DEATH_RUN = [618, 615, 612, 610, 607, 603, 603, 602, 596, 594,
                       588, 586, 586, 580, 578, 572, 571, 568, 564, 563,
                       556, 555, 555, 548, 547, 540, 538, 538, 532, 530, 524]


def test_scroll_velocity_not_biased_high_by_quantization_zeros():
    """Regression for the 2026-07-06 14:40 death: filtering per frame PAIR
    against SCROLL_V_MIN drops the quantization-zero steps but keeps the
    fast ones, biasing the estimate ~+30% (105.7 reported vs ~80 true) —
    which widened the pit window to [27,40], fired the jump at dist 39, and
    landed the cart 16px inside the pit's far lip. The windowed estimator
    must stay near the true speed on the same track."""
    sv = ScrollVelocity()
    _feed(sv, PIT_TRACK_DEATH_RUN)
    v = sv.velocity()
    assert v is not None
    assert v <= 90          # the biased estimator read ~100+ here
    assert 65 <= v          # sanity: it's still measuring real scroll


def test_scale_window_static_until_warmed_then_scales():
    assert scale_window(20, 30, None, SCROLL_V_REF_PIT) == (20, 30)
    lo, hi = scale_window(20, 30, 117.0, SCROLL_V_REF_PIT)
    assert (lo, hi) == (30, 44)  # 20*117/79, 30*117/79 rounded
    # At the reference speed the window is unchanged.
    assert scale_window(20, 30, SCROLL_V_REF_PIT, SCROLL_V_REF_PIT) == (20, 30)


def test_scale_slam_dist_never_drops_below_static_floor():
    """find_next_terrain floors ore distance at SCAN_BUFFER_PX(=10): a
    scaled-down slam window below ~11 could never fire at all."""
    assert scale_slam_dist(11, None, SCROLL_V_REF_ORE) == 11
    assert scale_slam_dist(11, 60.0, SCROLL_V_REF_ORE) == 11    # slow: floor holds
    assert scale_slam_dist(11, 117.0, SCROLL_V_REF_ORE) == 15   # fast: scales up


def test_airborne_guard_replays_botrun_jump_arc():
    """Replay the real jump from botrun_20260615_012340: feed cart_y per
    frame through GroundedBaseline and assert the guard would suppress a
    click during the up-arc (frames 34-36) and allow one once grounded —
    i.e. the documented mid-arc slam is prevented, validated on real frames."""
    base = Path(__file__).parent.parent / "minigames" / "mining" / "assets" / "captures" / "botrun_20260615_012340"
    frames = sorted(glob.glob(str(base / "frame_*.png")))
    if not frames:
        import pytest
        pytest.skip("botrun frames not present")
    gb = GroundedBaseline()
    prior = None
    airborne_by_frame = {}
    for i, f in enumerate(frames[:42], start=1):
        img = cv2.imread(f)
        py = D._find_plank_top_y(img)
        det = D.find_cart_detailed(img, plank_y=py, prior=prior)
        if det:
            prior = det
        cart_y = det["center"][1] if det else None
        gb.update(cart_y)
        airborne_by_frame[i] = is_cart_airborne(cart_y, gb.baseline())
    # Grounded run-up (frames 5-33) is never airborne.
    assert not any(airborne_by_frame[i] for i in range(5, 34))
    # The up-arc peak (frames 35-36, cart_y ~133-137) is airborne -> suppressed.
    assert airborne_by_frame[35] is True
    assert airborne_by_frame[36] is True
