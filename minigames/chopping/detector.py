import cv2
import numpy as np

# OpenCV HSV: H in [0,179], S/V in [0,255]. Red wraps, so two ranges.
#
# The "pointer" is a leaf sprite that scrolls back and forth across the bar.
# Per community wisdom, the hitbox is the LEFT edge of the leaf — that's what
# we use for the zone lookup (not the leaf's center or rightmost column).
#
# LEAF_HSV is a guess; tune via chopping-calibrate. Leaf is some shade of
# green/brown — make sure the range doesn't overlap with the zone-green below
# or we'll false-detect the bar itself.
LEAF_HSV = ((35, 100, 60), (60, 255, 200))         # darker/duller green than zone
GREEN_HSV = ((40, 80, 80), (80, 255, 255))         # bright zone-green
GOLD_HSV = ((20, 120, 120), (35, 255, 255))
RED_HSV_LOW = ((0, 120, 80), (10, 255, 255))
RED_HSV_HIGH = ((170, 120, 80), (179, 255, 255))


def _mask(hsv: np.ndarray, low, high) -> np.ndarray:
    return cv2.inRange(hsv, np.array(low), np.array(high))


def _column_has_color(mask: np.ndarray, x: int, min_pixels: int = 2) -> bool:
    if x < 0 or x >= mask.shape[1]:
        return False
    return int(mask[:, x].sum() // 255) >= min_pixels


def _leftmost_column(mask: np.ndarray, min_pixels_per_col: int = 2) -> int | None:
    """Return the leftmost X where the mask has at least N pixels in its column.

    Used for the leaf's left edge — that's the click hitbox per community
    wisdom. Filtering by min_pixels avoids picking up isolated noise pixels.
    """
    cols = (mask > 0).sum(axis=0)
    qualifying = np.where(cols >= min_pixels_per_col)[0]
    if len(qualifying) == 0:
        return None
    return int(qualifying[0])


def analyze_bar(bar_frame: np.ndarray, leaf_frame: np.ndarray | None = None) -> tuple[int | None, str]:
    """Return (leaf_left_edge_x, zone_under_left_edge).

    leaf_frame is the strip ABOVE the bar where the leaf scrolls; it MUST be
    horizontally aligned with bar_frame (same left/right window-relative). The
    leaf's X is detected in leaf_frame and looked up against zones in bar_frame.

    If leaf_frame is None, falls back to looking for the leaf in bar_frame
    itself — for setups where leaf and bar overlap.

    zone is one of: 'green', 'gold', 'red', 'none'.
    """
    bar_bgr = cv2.cvtColor(bar_frame, cv2.COLOR_BGRA2BGR)
    bar_hsv = cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2HSV)

    leaf_source_hsv = bar_hsv
    if leaf_frame is not None:
        leaf_bgr = cv2.cvtColor(leaf_frame, cv2.COLOR_BGRA2BGR)
        leaf_source_hsv = cv2.cvtColor(leaf_bgr, cv2.COLOR_BGR2HSV)

    leaf_mask = _mask(leaf_source_hsv, *LEAF_HSV)
    leaf_x = _leftmost_column(leaf_mask)
    if leaf_x is None:
        return None, "none"

    green = _mask(bar_hsv, *GREEN_HSV)
    gold = _mask(bar_hsv, *GOLD_HSV)
    red = cv2.bitwise_or(_mask(bar_hsv, *RED_HSV_LOW), _mask(bar_hsv, *RED_HSV_HIGH))

    # Check zones in priority order: gold > green > red.
    if _column_has_color(gold, leaf_x):
        return leaf_x, "gold"
    if _column_has_color(green, leaf_x):
        return leaf_x, "green"
    if _column_has_color(red, leaf_x):
        return leaf_x, "red"
    return leaf_x, "none"
