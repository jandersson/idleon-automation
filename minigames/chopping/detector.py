import cv2
import numpy as np

# TODO: tune these HSV ranges against real game screenshots.
# OpenCV HSV: H in [0,179], S/V in [0,255]. Red wraps, so two ranges.
POINTER_HSV = ((0, 0, 200), (179, 40, 255))        # placeholder: bright/white pointer
GREEN_HSV = ((40, 80, 80), (80, 255, 255))
GOLD_HSV = ((20, 120, 120), (35, 255, 255))
RED_HSV_LOW = ((0, 120, 80), (10, 255, 255))
RED_HSV_HIGH = ((170, 120, 80), (179, 255, 255))


def _mask(hsv: np.ndarray, low, high) -> np.ndarray:
    return cv2.inRange(hsv, np.array(low), np.array(high))


def _column_has_color(mask: np.ndarray, x: int, min_pixels: int = 2) -> bool:
    if x < 0 or x >= mask.shape[1]:
        return False
    return int(mask[:, x].sum() // 255) >= min_pixels


def analyze_bar(bar_frame: np.ndarray) -> tuple[int | None, str]:
    """Return (pointer_x, zone_under_pointer).

    zone is one of: 'green', 'gold', 'red', 'none'.
    """
    bgr = cv2.cvtColor(bar_frame, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    pointer_mask = _mask(hsv, *POINTER_HSV)
    cols = pointer_mask.sum(axis=0)
    if cols.max() == 0:
        return None, "none"
    pointer_x = int(cols.argmax())

    green = _mask(hsv, *GREEN_HSV)
    gold = _mask(hsv, *GOLD_HSV)
    red = cv2.bitwise_or(_mask(hsv, *RED_HSV_LOW), _mask(hsv, *RED_HSV_HIGH))

    # Check zones in priority order: gold > green > red.
    if _column_has_color(gold, pointer_x):
        return pointer_x, "gold"
    if _column_has_color(green, pointer_x):
        return pointer_x, "green"
    if _column_has_color(red, pointer_x):
        return pointer_x, "red"
    return pointer_x, "none"
