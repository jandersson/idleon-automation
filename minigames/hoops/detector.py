import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.templates import match_multiscale_center

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
    """Find the basketball hoop in the right half of the frame, scale-invariant."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    h, w = bgr.shape[:2]
    template = _load("hoop.png")
    center, val, _scale = match_multiscale_center(bgr, template, region=(w // 2, 0, w, h))
    if val < threshold:
        return None, val
    return center, val


def find_platform(
    frame: np.ndarray, threshold: float = 0.5
) -> tuple[tuple[int, int] | None, float]:
    """Find the character's platform in the left half of the frame, scale-invariant.

    Threshold lowered from 0.7 since multi-scale matching at off-tuned scales
    naturally peaks lower; 0.5 still discriminates the platform from background.
    """
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    h, w = bgr.shape[:2]
    template = _load("platform.png")
    center, val, _scale = match_multiscale_center(bgr, template, region=(0, 0, w // 2, h))
    if val < threshold:
        return None, val
    return center, val


# Orange basketball. HSV bounds are deliberately broad (catches the ball even
# under different background colors); blob-area limits exclude both small noise
# and the merged ball+rim region (which would be larger than 2000px²).
BALL_HSV_LOWER = np.array([5, 120, 120])
BALL_HSV_UPPER = np.array([20, 255, 255])
BALL_MIN_AREA = 10  # lowered for motion-masked detection: only the ball's
                    # leading/trailing edge survives the motion filter (since
                    # the previous ball position was elsewhere and the rim is
                    # static), so the surviving blob is smaller than the raw
                    # color blob.
BALL_MAX_AREA = 2500


def find_game_over(
    frame: np.ndarray, threshold: float = 0.7
) -> tuple[bool, float]:
    """Detect the end-of-trial 'Game over!' screen via multi-scale template match."""
    return _find_top_text(frame, "game_over.png", threshold)


def find_game_prompt(
    frame: np.ndarray, threshold: float = 0.65
) -> tuple[bool, float]:
    """Detect the 'Make a shot to start the game!' prompt that appears before
    the game has begun. While this is on screen, the next click only dismisses
    the prompt; it doesn't count toward score. Slightly lower threshold than
    game_over since the prompt has more pixel-art noise around it.
    """
    return _find_top_text(frame, "game_prompt.png", threshold)


def _find_top_text(frame: np.ndarray, template_name: str, threshold: float) -> tuple[bool, float]:
    path = ASSETS / template_name
    if not path.exists():
        return False, 0.0
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    template = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if template is None:
        return False, 0.0
    _, val, _scale = match_multiscale_center(bgr, template)
    return val >= threshold, val


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
    """Detect if the score-region crop's DIGITS changed.

    Both crops are binarized via Otsu before comparing. This collapses the
    animated background (twinkling stars, parallax) to a single value, so
    only high-contrast digit pixels contribute to the diff. Result: real
    score changes give large diffs (~10-50) while background motion gives
    near zero, distinguishing them cleanly.
    """
    if before.shape != after.shape:
        return True, 255.0
    _, b_bin = cv2.threshold(before, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, a_bin = cv2.threshold(after, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    diff = cv2.absdiff(b_bin, a_bin).astype(np.float32)
    mean_diff = float(diff.mean())
    return mean_diff > threshold, mean_diff


def find_ball(
    frame: np.ndarray,
    search_x0: int,
    search_y0: int,
    search_x1: int,
    search_y1: int,
    prev_frame: np.ndarray | None = None,
    motion_threshold: float = 12.0,
) -> tuple[int, int] | None:
    """HSV-mask for orange in airspace, optionally restricted to MOVING pixels.

    The static rim/backboard is also orange, so the prior HSV-only detection
    was finding the rim's centroid as the "ball." Passing prev_frame masks the
    color result by frame-to-frame motion: only pixels that are orange AND
    changed between this frame and the previous one count. Result: rim is
    filtered out automatically since it doesn't move; only the airborne ball
    survives both filters.

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
    color_mask = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)

    mask = color_mask
    if prev_frame is not None:
        prev_bgr = cv2.cvtColor(prev_frame, cv2.COLOR_BGRA2BGR)
        if prev_bgr.shape == bgr.shape:
            prev_crop = prev_bgr[y0:y1, x0:x1]
            diff = cv2.absdiff(crop, prev_crop).astype(np.float32).mean(axis=2)
            motion_mask = (diff > motion_threshold).astype(np.uint8) * 255
            mask = cv2.bitwise_and(color_mask, motion_mask)

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
