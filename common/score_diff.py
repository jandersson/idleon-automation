"""Pixel-diff-based make/miss detection for minigames with a numeric score.

Both hoops and darts (and any future minigame with a "Score: N" readout) use
the same trick: capture the score-number crop before and after a shot, run an
Otsu-binarized diff, and call it a make if the binary diff exceeds a small
threshold. Otsu collapses background animation to a constant; only digit
changes contribute to the diff.

Used in concert with `common.regions.get_region(_HERE, "score", win_w, win_h)`.
"""
import cv2
import numpy as np


def score_region(
    frame: np.ndarray,
    region_left: int,
    region_top: int,
    region_width: int,
    region_height: int,
) -> np.ndarray:
    """Return a grayscale crop of the score readout, ready for score_changed."""
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
    only high-contrast digit pixels contribute to the diff. Real digit changes
    give large diffs (~10-50); background motion gives near zero.
    """
    if before.shape != after.shape:
        return True, 255.0
    _, b_bin = cv2.threshold(before, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, a_bin = cv2.threshold(after, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    diff = cv2.absdiff(b_bin, a_bin).astype(np.float32)
    mean_diff = float(diff.mean())
    return mean_diff > threshold, mean_diff
