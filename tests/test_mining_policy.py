"""Tests for minigames/mining/policy.py — the pure jump-fire predicate and
the grounded-baseline / airborne guard."""
import glob
from pathlib import Path

import cv2

from minigames.mining.policy import (
    GroundedBaseline,
    JUMP_GROUNDED_EPS_PX,
    is_cart_airborne,
    should_jump,
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
