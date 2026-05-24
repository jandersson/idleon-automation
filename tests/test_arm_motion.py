"""Tests for minigames.darts.arm_motion.compute_arm_centroid."""
import numpy as np

from minigames.darts.arm_motion import (
    MIN_AREA,
    WHITE_HSV_HIGH,
    WHITE_HSV_LOW,
    compute_arm_centroid,
)


def _frame(h: int = 100, w: int = 100, fill=(50, 30, 30)) -> np.ndarray:
    return np.full((h, w, 3), fill, dtype=np.uint8)


def test_no_prev_frame_returns_none():
    cur = _frame()
    centroid, area = compute_arm_centroid(cur, None)
    assert centroid is None
    assert area == 0


def test_mismatched_shapes_returns_none():
    cur = _frame(100, 100)
    prev = _frame(50, 50)
    centroid, area = compute_arm_centroid(cur, prev)
    assert centroid is None
    assert area == 0


def test_no_motion_returns_none_with_zero_area():
    """Two identical frames → no motion → no arm pixels found."""
    frame = _frame()
    centroid, area = compute_arm_centroid(frame, frame)
    assert centroid is None
    assert area == 0


def test_white_pixels_moved_produces_centroid():
    """A patch of white pixels that appeared since the last frame
    contributes to the centroid."""
    prev = _frame()  # all dark
    cur = _frame()
    # Plant a 15x15 block of player-white (high V, low S) at y=40-55
    cur[40:55, 30:45] = (255, 255, 255)
    centroid, area = compute_arm_centroid(cur, prev)
    assert centroid is not None
    # Centroid y should be in the middle of the block (~47)
    assert 40 <= centroid <= 55
    assert area >= MIN_AREA


def test_motion_without_white_is_ignored():
    """Non-white motion (e.g. background color change) shouldn't count
    as arm pixels — the bot's discriminator depends on isolating the
    player sprite, not just any moving pixels."""
    prev = _frame(fill=(50, 30, 30))
    cur = _frame(fill=(50, 30, 30))
    # Plant motion in a NON-white colored region (yellow, high S)
    cur[40:55, 30:45] = (0, 200, 200)
    centroid, area = compute_arm_centroid(cur, prev)
    # White mask filters this out
    assert centroid is None


def test_small_motion_blob_below_min_area_returns_none_centroid():
    """A few stray white pixels (UI antialiasing, noise) shouldn't
    register as the arm — the centroid would be unreliable."""
    prev = _frame()
    cur = _frame()
    cur[10:12, 10:12] = (255, 255, 255)  # 4 white pixels — well below MIN_AREA
    centroid, area = compute_arm_centroid(cur, prev)
    assert centroid is None
    assert area < MIN_AREA


def test_centroid_moves_with_blob_position():
    """Sanity: centroid is sensitive to where the moving pixels are."""
    prev = _frame()
    high = _frame()
    high[20:35, 30:45] = (255, 255, 255)  # blob near top
    low = _frame()
    low[70:85, 30:45] = (255, 255, 255)   # blob near bottom
    high_cy, _ = compute_arm_centroid(high, prev)
    low_cy, _ = compute_arm_centroid(low, prev)
    assert high_cy is not None and low_cy is not None
    assert high_cy < low_cy


def test_hsv_bounds_constants_have_expected_shape():
    """Guard against accidental changes to the HSV palette — these
    bounds are calibrated to the player sprite's white."""
    assert WHITE_HSV_LOW.shape == (3,)
    assert WHITE_HSV_HIGH.shape == (3,)
    assert WHITE_HSV_LOW[2] < WHITE_HSV_HIGH[2]  # V band non-empty
