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


def find_release_pose(
    frame: np.ndarray, threshold: float = 0.7
) -> tuple[tuple[int, int] | None, float]:
    """Template-match the player's hand+dart in the release angle.

    The hand sweeps periodically through a `)` arc. The template is captured
    at the desired release angle (typically arm horizontal-right toward the
    target), so matchTemplate confidence peaks once per arc cycle when the
    hand is at that angle.

    Returns (center_xy_in_frame, confidence) or (None, confidence).
    """
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    template = _load("release.png")
    result = cv2.matchTemplate(bgr, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < threshold:
        return None, max_val
    th, tw = template.shape[:2]
    cx = max_loc[0] + tw // 2
    cy = max_loc[1] + th // 2
    return (cx, cy), max_val


def score_region(
    frame: np.ndarray,
    region_left: int,
    region_top: int,
    region_width: int,
    region_height: int,
) -> np.ndarray:
    """Grayscale crop of the score readout, for diff-based change detection.

    Same approach as hoops/detector.py — kept here as a per-minigame copy so
    the darts bot can have its own score region without coupling to hoops.
    """
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    h, w = bgr.shape[:2]
    x0 = max(0, region_left)
    y0 = max(0, region_top)
    x1 = min(w, region_left + region_width)
    y1 = min(h, region_top + region_height)
    crop = bgr[y0:y1, x0:x1]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def score_changed(before: np.ndarray, after: np.ndarray, threshold: float = 5.0) -> tuple[bool, float]:
    if before.shape != after.shape:
        return True, 255.0
    diff = cv2.absdiff(before, after).astype(np.float32)
    mean_diff = float(diff.mean())
    return mean_diff > threshold, mean_diff
