"""Tests for minigames.darts.arm_motion."""
import numpy as np

from minigames.darts.arm_motion import (
    MAX_VY_DT_S,
    MIN_AREA,
    WHITE_HSV_HIGH,
    WHITE_HSV_LOW,
    centroid_vy_px_s,
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


def test_vy_is_time_denominated():
    """px/second, invariant to poll cadence: the same displacement over
    half the time is twice the velocity."""
    assert centroid_vy_px_s(364, 374, 0.25) == -40
    assert centroid_vy_px_s(364, 374, 0.125) == -80
    assert centroid_vy_px_s(374, 364, 0.25) == 40


def test_vy_bridges_dropouts_by_elapsed_time():
    """A masked-out poll between two valid centroids just lengthens dt
    (#42): displacement over the real elapsed time is still a velocity.
    Values from the #42 diagnosis — session 2026-06-10T19:48 throw 6
    fired at cen_y=364 with the last valid centroid 374 ~0.5s back."""
    assert centroid_vy_px_s(364, 374, 0.5) == -20


def test_vy_none_beyond_max_dt():
    """Too long since the last valid centroid: the arm has moved
    unobserved for too long for a velocity to mean anything."""
    assert centroid_vy_px_s(364, 374, MAX_VY_DT_S + 0.01) is None
    assert centroid_vy_px_s(364, 374, MAX_VY_DT_S) is not None


def test_vy_none_when_either_centroid_missing():
    assert centroid_vy_px_s(None, 374, 0.25) is None
    assert centroid_vy_px_s(364, None, 0.25) is None


def test_vy_rejects_nonpositive_dt():
    """dt <= 0 is a bookkeeping bug upstream — refuse to divide."""
    assert centroid_vy_px_s(364, 374, 0.0) is None
    assert centroid_vy_px_s(364, 374, -0.25) is None


def test_hsv_bounds_constants_have_expected_shape():
    """Guard against accidental changes to the HSV palette — these
    bounds are calibrated to the player sprite's white."""
    assert WHITE_HSV_LOW.shape == (3,)
    assert WHITE_HSV_HIGH.shape == (3,)
    assert WHITE_HSV_LOW[2] < WHITE_HSV_HIGH[2]  # V band non-empty
