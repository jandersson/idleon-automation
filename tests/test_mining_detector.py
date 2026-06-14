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
    CART_SCALES,
    CART_TRACK_SCALE_STEPS,
    _find_plank_top_y,
    _load_cart_templates,
    _scales_near,
    _scan_plank_ore,
    _scan_plank_pits,
    find_cart,
    find_cart_detailed,
    find_next_terrain,
    pts_score_region,
    read_score,
    read_score_from_crop,
    score_crop,
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


def test_pts_score_region_is_plank_relative_below_left():
    # The PTS digits sit below-left of the plank, right-anchored near
    # plank_x0 (#6). Region is None without a located plank.
    assert pts_score_region(None, (100, 400)) is None
    assert pts_score_region(160, None) is None
    box = pts_score_region(160, (100, 400))
    x0, y0, x1, y1 = box
    assert y0 > 160 and y1 > y0          # below the plank
    assert x1 <= 100 + 5 and x0 < 100    # ends at/just past plank_x0, extends left
    assert x1 > x0


def test_read_score_none_without_plank():
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    assert read_score(frame, None, None) is None


def test_score_crop_matches_pts_region_box():
    # The crop returned for digit capture / OCR must be exactly the
    # pts_score_region box (when the box is fully inside the frame).
    frame = _frame_with_plank()
    plank_range = (100, 400)
    box = pts_score_region(PLANK_Y, plank_range)
    x0, y0, x1, y1 = box
    crop = score_crop(frame, PLANK_Y, plank_range)
    assert crop is not None
    assert crop.shape[:2] == (y1 - y0, x1 - x0)


def test_score_crop_none_without_plank():
    frame = _frame_with_plank()
    assert score_crop(frame, None, (100, 400)) is None
    assert score_crop(frame, PLANK_Y, None) is None


def test_read_score_equals_crop_then_read():
    # read_score must be exactly read_score_from_crop(score_crop(...)) so
    # the main loop can crop once and feed the same pixels to both the OCR
    # and the digit capturer.
    frame = _frame_with_plank()
    plank_range = (100, 400)
    crop = score_crop(frame, PLANK_Y, plank_range)
    assert read_score(frame, PLANK_Y, plank_range) == read_score_from_crop(crop)


def test_read_score_from_crop_none_input():
    assert read_score_from_crop(None) is None


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


def _frame_with_pasted_cart(paste_x=400):
    """A plank frame with cart template 0 pasted on it; returns (frame, template)."""
    templates = _load_cart_templates()
    _name, template = templates[0]
    th, tw = template.shape[:2]
    frame = _frame_with_plank()
    paste_y = PLANK_Y - th
    frame[paste_y:paste_y + th, paste_x:paste_x + tw] = template
    return frame, template


def test_find_cart_detailed_returns_match_fields():
    frame, template = _frame_with_pasted_cart()
    det = find_cart_detailed(frame)
    assert det is not None
    assert set(det) >= {"center", "half_width", "scale", "template", "score"}
    assert det["score"] >= 0.8
    assert det["half_width"] > 0
    # center x lands within the pasted template span
    assert 400 <= det["center"][0] <= 400 + template.shape[1]


def test_find_cart_detailed_prior_agrees_with_full():
    """The narrow prior-anchored search returns the same center as the
    full search (the loop's whole correctness premise)."""
    frame, _ = _frame_with_pasted_cart()
    full = find_cart_detailed(frame)
    tracked = find_cart_detailed(frame, prior=full)
    assert tracked is not None
    assert abs(tracked["center"][0] - full["center"][0]) <= 6
    assert abs(tracked["center"][1] - full["center"][1]) <= 8


def test_find_cart_detailed_stale_prior_falls_back_to_full():
    """A prior pointing far from the cart must not blind the detector —
    the narrow miss falls back to a full-frame search."""
    frame, _ = _frame_with_pasted_cart(paste_x=400)
    stale = {"center": (50, PLANK_Y), "scale": 1.0}
    det = find_cart_detailed(frame, prior=stale)
    assert det is not None
    assert 400 <= det["center"][0] <= 500


def test_find_next_terrain_uses_passed_cart_right_and_plank_y():
    """When the hot path passes plank_y + cart_right, distance is measured
    from that cart_right and no recomputation changes the result."""
    frame = _frame_with_plank()
    frame[PLANK_Y:PLANK_Y + 12, 500:560] = 0  # 60px pit
    res = find_next_terrain(frame, (300, PLANK_Y), plank_y=PLANK_Y, cart_right=350)
    assert res == {"kind": "pit", "x": 500, "distance_px": 150}


def test_scales_near_window():
    assert _scales_near(None) == CART_SCALES          # unknown → full sweep
    assert _scales_near(0.123) == CART_SCALES          # not a known step → full sweep
    near_one = _scales_near(1.0)
    assert 1.0 in near_one and len(near_one) == 2 * CART_TRACK_SCALE_STEPS + 1
    # endpoints clamp instead of wrapping
    assert _scales_near(CART_SCALES[0])[0] == CART_SCALES[0]
