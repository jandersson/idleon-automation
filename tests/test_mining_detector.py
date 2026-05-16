"""Tests for mining detector — plank-finding + pit/ore scanners.

Synthetic frames only. Real-game-frame validation lives in
analysis/pit_detect_overlay.png against the captured traces.

Note: find_next_terrain is not unit-tested at the integration level yet
because cart x is dynamic (varies by player world position) and a
proper find_cart implementation is blocked on issue #3. The scanners
take an `x_start` arg so they can be tested in isolation.
"""
import cv2
import numpy as np

from minigames.mining.detector import (
    ORE_MIN_WIDTH,
    PIT_MIN_WIDTH,
    PLANK_X0_FRAC,
    PLANK_X1_FRAC,
    _find_plank_top_y,
    _scan_plank_ore,
    _scan_plank_pits,
)

# Synthetic-frame parameters — pick a window size and plank position the
# tests own, so the auto-detection has something concrete to find.
W, H = 960, 572
PLANK_Y = 160  # plank top row
PLANK_X0 = int(PLANK_X0_FRAC * W)
PLANK_X1 = int(PLANK_X1_FRAC * W)


def _frame_with_plank() -> np.ndarray:
    """A black frame with a bright tan plank band at y=PLANK_Y..PLANK_Y+12."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    # BGR for tan/wood — converts to HSV roughly (H=12, S=140, V=200) which
    # satisfies the plank-signature thresholds.
    frame[PLANK_Y:PLANK_Y + 12, PLANK_X0:PLANK_X1] = (40, 120, 200)
    return frame


def test_find_plank_top_y_locates_synthetic_plank():
    frame = _frame_with_plank()
    assert _find_plank_top_y(frame) == PLANK_Y


def test_find_plank_top_y_returns_none_on_empty_frame():
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    assert _find_plank_top_y(frame) is None


def test_scan_plank_pits_finds_single_pit():
    frame = _frame_with_plank()
    # Punch a 10-px wide pit on the plank surface.
    frame[PLANK_Y:PLANK_Y + 12, 500:510] = 0
    pits = _scan_plank_pits(frame, PLANK_Y)
    assert pits == [(500, 510)]


def test_scan_plank_pits_respects_x_start():
    frame = _frame_with_plank()
    frame[PLANK_Y:PLANK_Y + 12, 400:410] = 0
    frame[PLANK_Y:PLANK_Y + 12, 600:610] = 0
    # Skip past the first pit — only the second should be reported.
    pits = _scan_plank_pits(frame, PLANK_Y, x_start=500)
    assert pits == [(600, 610)]


def test_scan_plank_pits_ignores_narrow_noise():
    frame = _frame_with_plank()
    # Pit narrower than PIT_MIN_WIDTH should be dropped.
    frame[PLANK_Y:PLANK_Y + 12, 500:500 + PIT_MIN_WIDTH - 1] = 0
    assert _scan_plank_pits(frame, PLANK_Y) == []


def test_scan_plank_ore_finds_bright_protrusion():
    frame = _frame_with_plank()
    # Ore: bright tan pixels in the band ABOVE the plank top.
    # Plank-hue + bright V satisfies _scan_plank_ore criteria.
    frame[PLANK_Y - 8:PLANK_Y, 500:520] = (40, 120, 220)
    ores = _scan_plank_ore(frame, PLANK_Y)
    assert ores == [(500, 520)]


def test_scan_plank_ore_ignores_narrow_noise():
    frame = _frame_with_plank()
    frame[PLANK_Y - 8:PLANK_Y, 500:500 + ORE_MIN_WIDTH - 1] = (40, 120, 220)
    assert _scan_plank_ore(frame, PLANK_Y) == []


def test_scan_plank_ore_respects_x_start():
    frame = _frame_with_plank()
    frame[PLANK_Y - 8:PLANK_Y, 400:420] = (40, 120, 220)
    frame[PLANK_Y - 8:PLANK_Y, 600:620] = (40, 120, 220)
    ores = _scan_plank_ore(frame, PLANK_Y, x_start=500)
    assert ores == [(600, 620)]
