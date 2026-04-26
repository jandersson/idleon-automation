"""Catching minigame detectors.

Two things to find each frame:
- The fly (player) — its (x, y) inside the play region.
- The next hoop's gap — vertical center band of the next ring ahead of the fly.
"""
from pathlib import Path

import cv2
import numpy as np

ASSETS = Path(__file__).parent / "assets"

# Golden rings — saturated yellow/orange. Tune via captures.
RING_HSV_LOWER = np.array([15, 120, 140])
RING_HSV_UPPER = np.array([35, 255, 255])
RING_MIN_AREA = 200    # ignore noise (small particles, distant rings)
RING_MAX_AREA = 50000  # ignore very-large gold UI elements


def find_fly(frame: np.ndarray) -> tuple[int, int] | None:
    """Template-match assets/fly.png. Returns (x, y) center or None.

    Auto-extract a fly template via catching-extract-fly after running
    catching-capture during a real attempt.
    """
    fly_path = ASSETS / "fly.png"
    if not fly_path.exists():
        return None
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    template = cv2.imread(str(fly_path), cv2.IMREAD_COLOR)
    if template is None:
        return None
    if bgr.shape[0] < template.shape[0] or bgr.shape[1] < template.shape[1]:
        return None
    result = cv2.matchTemplate(bgr, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < 0.6:
        return None
    th, tw = template.shape[:2]
    return (max_loc[0] + tw // 2, max_loc[1] + th // 2)


def find_next_gap(frame: np.ndarray, fly_pos: tuple[int, int] | None) -> tuple[int, int] | None:
    """Find the next gold ring ahead of the fly; return (top_y, bottom_y) of
    its inner hole — the band the fly should aim through.

    Approach: HSV-mask gold pixels, find connected components, pick the
    leftmost component whose center is to the RIGHT of fly_x (the next ring
    in scrolling order). Return its bounding box's vertical bounds.
    """
    if fly_pos is None:
        return None
    fly_x, _fly_y = fly_pos

    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, RING_HSV_LOWER, RING_HSV_UPPER)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < RING_MIN_AREA or area > RING_MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cx = x + w // 2
        if cx <= fly_x:
            continue  # ring already passed
        candidates.append((cx, x, y, w, h))

    if not candidates:
        return None

    # Pick the ring with smallest cx > fly_x (i.e. the next one ahead).
    candidates.sort()
    _, x, y, w, h = candidates[0]
    return (y, y + h)
