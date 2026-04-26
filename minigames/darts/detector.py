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


def find_release_pose(
    frame: np.ndarray, threshold: float = 0.6
) -> tuple[tuple[int, int] | None, float]:
    """Template-match the player's hand+dart in the release angle, scale-invariant.

    The hand sweeps periodically through a `)` arc. The template is captured
    at the desired release angle, so matchTemplate confidence peaks once per
    arc cycle when the hand is at that angle. Multi-scale matching means the
    same template works whether the user resizes the game window.

    Threshold lowered from 0.7 to 0.6 since multi-scale tries off-tuned scales
    and naturally peaks lower; 0.6 still discriminates the release angle from
    other arm positions.
    """
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    template = _load("release.png")
    center, val, _scale = match_multiscale_center(bgr, template)
    if val < threshold:
        return None, val
    return center, val


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
