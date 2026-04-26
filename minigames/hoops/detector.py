import cv2
import numpy as np
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"


def _load(name: str) -> np.ndarray:
    path = ASSETS / name
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Template not found: {path}")
    return img


def _match_in_region(
    frame_bgr: np.ndarray,
    template: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    threshold: float,
) -> tuple[tuple[int, int] | None, float]:
    """Template-match `template` within frame_bgr[y0:y1, x0:x1].

    Returns (center_xy_in_full_frame, confidence) or (None, confidence).
    """
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.shape[0] < template.shape[0] or crop.shape[1] < template.shape[1]:
        return None, 0.0

    result = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < threshold:
        return None, max_val

    th, tw = template.shape[:2]
    cx = x0 + max_loc[0] + tw // 2
    cy = y0 + max_loc[1] + th // 2
    return (cx, cy), max_val


def find_hoop(
    frame: np.ndarray, threshold: float = 0.35
) -> tuple[tuple[int, int] | None, float]:
    """Find the basketball hoop — searched in the right half of the frame."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    h, w = bgr.shape[:2]
    template = _load("hoop.png")
    return _match_in_region(bgr, template, w // 2, 0, w, h, threshold)


def find_platform(
    frame: np.ndarray, threshold: float = 0.7
) -> tuple[tuple[int, int] | None, float]:
    """Find the character's platform — searched in the left half of the frame."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    h, w = bgr.shape[:2]
    template = _load("platform.png")
    return _match_in_region(bgr, template, 0, 0, w // 2, h, threshold)


# Orange basketball — defaults are a guess; tune via hoops-ball-calibrate.
BALL_HSV_LOWER = np.array([5, 120, 120])
BALL_HSV_UPPER = np.array([20, 255, 255])
BALL_MIN_AREA = 30
BALL_MAX_AREA = 800


def find_game_over(
    frame: np.ndarray, threshold: float = 0.7
) -> tuple[bool, float]:
    """Detect the end-of-trial 'Game over!' screen via template match.

    Returns (detected, confidence). If the template file doesn't exist yet,
    returns (False, 0.0) silently so the bot runs even before calibration.
    """
    path = ASSETS / "game_over.png"
    if not path.exists():
        return False, 0.0
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    template = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if template is None:
        return False, 0.0
    if bgr.shape[0] < template.shape[0] or bgr.shape[1] < template.shape[1]:
        return False, 0.0
    result = cv2.matchTemplate(bgr, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val >= threshold, max_val


def score_region(
    frame: np.ndarray,
    region_left: int,
    region_top: int,
    region_width: int,
    region_height: int,
) -> np.ndarray:
    """Return a grayscale crop of the score readout, for diff-based change detection."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    h, w = bgr.shape[:2]
    x0 = max(0, region_left)
    y0 = max(0, region_top)
    x1 = min(w, region_left + region_width)
    y1 = min(h, region_top + region_height)
    crop = bgr[y0:y1, x0:x1]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def score_changed(before: np.ndarray, after: np.ndarray, threshold: float = 3.0) -> tuple[bool, float]:
    """Mean absolute pixel difference between two crops; True if above threshold.

    Returns (changed, mean_diff). The mean_diff is in 0-255 grayscale units;
    threshold ~5 reliably distinguishes a digit changing from background noise.
    """
    if before.shape != after.shape:
        return True, 255.0
    diff = cv2.absdiff(before, after).astype(np.float32)
    mean_diff = float(diff.mean())
    return mean_diff > threshold, mean_diff


def find_ball(
    frame: np.ndarray,
    search_x0: int,
    search_y0: int,
    search_x1: int,
    search_y1: int,
) -> tuple[int, int] | None:
    """HSV-mask for orange in the given window-relative airspace.

    Returns ball center (x, y) in the full frame, or None.
    """
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    h, w = bgr.shape[:2]
    x0 = max(0, search_x0)
    y0 = max(0, search_y0)
    x1 = min(w, search_x1)
    y1 = min(h, search_y1)
    if x1 <= x0 or y1 <= y0:
        return None

    crop = bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < BALL_MIN_AREA or area > BALL_MAX_AREA:
            continue
        if area > best_area:
            best_area = area
            best = c
    if best is None:
        return None
    M = cv2.moments(best)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"]) + x0
    cy = int(M["m01"] / M["m00"]) + y0
    return (cx, cy)
