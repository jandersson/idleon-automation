"""Tests for mining detector — plank pit scanner + find_next_terrain.

Synthetic frames only — real-game-frame validation lives in
analysis/pit_detect_overlay.png against trace_20260515_220235.
"""
import numpy as np

from minigames.mining.detector import (
    CART_RIGHT_X,
    PIT_MIN_WIDTH,
    PIT_V_MAX,
    PLANK_X0,
    PLANK_X1,
    PLANK_Y0,
    PLANK_Y1,
    _scan_plank_pits,
    find_next_terrain,
)


def _blank_frame(h: int = 600, w: int = 960) -> np.ndarray:
    """Black BGR frame the same height as the calibration window."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _paint_plank(frame: np.ndarray, x0: int, x1: int) -> None:
    """Fill the plank band x[x0..x1] with a bright-enough tan to NOT trigger
    the pit threshold. V threshold is PIT_V_MAX=80, so use V=200."""
    # BGR for orange-ish tan: (40, 100, 200) → V≈200
    frame[PLANK_Y0:PLANK_Y1, x0:x1] = (40, 100, 200)


def test_scan_finds_single_pit():
    frame = _blank_frame()
    _paint_plank(frame, PLANK_X0, PLANK_X1)
    # Punch a 10-px-wide pit in the middle of the plank.
    frame[PLANK_Y0:PLANK_Y1, 500:510] = 0
    pits = _scan_plank_pits(frame)
    assert pits == [(500, 510)]


def test_scan_finds_multiple_pits():
    frame = _blank_frame()
    _paint_plank(frame, PLANK_X0, PLANK_X1)
    frame[PLANK_Y0:PLANK_Y1, 400:410] = 0
    frame[PLANK_Y0:PLANK_Y1, 600:615] = 0
    pits = _scan_plank_pits(frame)
    assert pits == [(400, 410), (600, 615)]


def test_scan_ignores_narrow_noise():
    """Single-pixel dark specks shouldn't register as pits."""
    frame = _blank_frame()
    _paint_plank(frame, PLANK_X0, PLANK_X1)
    # Pit narrower than PIT_MIN_WIDTH (5) — should be dropped.
    frame[PLANK_Y0:PLANK_Y1, 500:503] = 0
    pits = _scan_plank_pits(frame)
    assert pits == []


def test_scan_ignores_dark_outside_plank_x_range():
    """Cave wall pixels outside PLANK_X0..PLANK_X1 shouldn't be reported."""
    frame = _blank_frame()
    # Don't paint a plank — entire scan band is dark. Scanner should
    # report nothing inside the plank x range (because no plank there
    # either, but the slice itself is empty/dark, so no runs would form
    # *within* PLANK_X0..PLANK_X1... actually a uniformly dark band
    # IS one big run). Test instead: a dark blob outside the range
    # must not be reported.
    _paint_plank(frame, PLANK_X0, PLANK_X1)
    # Dark blob outside the scanned x range — paint OVER the plank to
    # the LEFT of PLANK_X0.
    frame[PLANK_Y0:PLANK_Y1, 100:200] = 0  # entirely outside scan
    pits = _scan_plank_pits(frame)
    assert pits == []


def test_find_next_terrain_picks_nearest_ahead():
    frame = _blank_frame()
    _paint_plank(frame, PLANK_X0, PLANK_X1)
    # Two pits ahead of the cart, one closer.
    near_x = CART_RIGHT_X + 50
    far_x = CART_RIGHT_X + 200
    frame[PLANK_Y0:PLANK_Y1, near_x:near_x + 10] = 0
    frame[PLANK_Y0:PLANK_Y1, far_x:far_x + 10] = 0
    result = find_next_terrain(frame, cart=(325, 312))
    assert result == {"kind": "pit", "x": near_x, "distance_px": 50}


def test_find_next_terrain_ignores_pits_behind_cart():
    frame = _blank_frame()
    _paint_plank(frame, PLANK_X0, PLANK_X1)
    # Pit to the LEFT of the cart — cart already passed it, irrelevant.
    behind_x = PLANK_X0 + 10
    frame[PLANK_Y0:PLANK_Y1, behind_x:behind_x + 10] = 0
    assert find_next_terrain(frame, cart=(325, 312)) is None


def test_find_next_terrain_returns_none_with_no_cart():
    frame = _blank_frame()
    assert find_next_terrain(frame, cart=None) is None
