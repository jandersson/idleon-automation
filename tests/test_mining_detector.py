"""Tests for mining detector — plank-finding, pit/ore scanners, cart matching.

Synthetic frames only. Real-game-frame validation lives in
analysis/pit_detect_overlay.png against the captured traces.
"""
from pathlib import Path

import cv2
import numpy as np

from minigames.mining.detector import (
    ORE_MIN_WIDTH,
    PIT_MIN_WIDTH,
    PLANK_X0_FRAC,
    PLANK_X1_FRAC,
    _find_plank_top_y,
    _load_cart_templates,
    _scan_plank_ore,
    _scan_plank_pits,
    find_cart,
)

ASSETS_DIR = Path(__file__).parent.parent / "minigames" / "mining" / "assets"

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


def test_load_cart_templates_finds_assets():
    templates = _load_cart_templates()
    assert len(templates) >= 1, "expected at least one cart template in assets/cart_*.png"
    for _name, t in templates:
        assert t.ndim == 3 and t.shape[2] == 3


def test_find_cart_locates_pasted_template():
    """Paste a cart template onto a synthetic plank frame and verify
    find_cart returns a center INSIDE the pasted region. (Not stricter
    than that because multiple templates exist — a smaller template may
    match a sub-region of the pasted larger one with even higher
    confidence, which is fine.)"""
    templates = _load_cart_templates()
    name, template = templates[0]
    th, tw = template.shape[:2]

    frame = _frame_with_plank()
    paste_x = 400
    paste_y = PLANK_Y - th
    frame[paste_y:paste_y + th, paste_x:paste_x + tw] = template

    found = find_cart(frame)
    assert found is not None, f"find_cart returned None for pasted {name}"
    cx, cy = found
    assert paste_x <= cx <= paste_x + tw, f"cx={cx} outside pasted x-range [{paste_x}, {paste_x+tw}]"
    assert paste_y - 5 <= cy <= paste_y + th + 5, f"cy={cy} outside pasted y-range"


def test_find_cart_returns_none_when_no_plank():
    """No plank → no cart search even attempted."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    assert find_cart(frame) is None
