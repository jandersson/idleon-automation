"""Detect the swinging arm by intersecting motion mask with player-white mask.

The dart bot's existing template-match approach gives a strong signal only
WHEN the dart is at the captured release angle — between match moments,
we have no continuous tracking of where the arm is. This module provides
that continuous signal:

1. Diff current frame against the previous → pixels that changed
2. AND with HSV mask for player-white (S<70, V>180) → only moving white pixels
3. Centroid + area of the result = where the arm is + how much is moving

Use case (#25): the bot fires at two distinct moments per swing cycle
(forward release → hit, top-of-swing apex → miss). Both produce the same
template-match conf in a single frame. Arm centroid y should differ —
arm is HIGH on screen at apex (low y value) vs LOWER at forward release.
"""
from __future__ import annotations

import cv2
import numpy as np


# HSV bounds for player-sprite white (matches auto_crop_release.py).
WHITE_HSV_LOW = np.array([0, 0, 180])
WHITE_HSV_HIGH = np.array([180, 70, 255])

# Per-pixel abs-diff threshold for "this pixel moved". 20 = ignore
# anti-aliasing noise and faint background drift, capture actual sprite
# changes. Same value the offline auto-crop tool uses.
MOTION_THRESHOLD = 20

# Minimum number of moving-white pixels to consider the centroid trustworthy.
# Below this we likely caught noise (a few stray pixels of UI animation),
# not an actual arm sweep.
MIN_AREA = 30

# How many polls back the dy computation may reach for its reference
# centroid. 2 bridges a single-poll dropout; longer gaps mean the arm
# velocity estimate is too stale to trust.
MAX_DY_GAP_POLLS = 2


def centroid_dy_per_poll(
    cur_y: int | None,
    prev_y: int | None,
    gap_polls: int,
    max_gap: int = MAX_DY_GAP_POLLS,
) -> int | None:
    """Per-poll vertical velocity of the arm centroid, bridged across
    short dropouts.

    `prev_y` is the last VALID centroid, captured `gap_polls` polls ago
    (1 = the immediately preceding poll). A slow-moving arm produces a
    small frame-diff, so `compute_arm_centroid` returns None (sub-
    MIN_AREA mask) exactly at the slow passes the dy-band gate selects
    for — requiring two consecutive valid polls blanked dy on ~1/3 of
    fires (#42). Dividing the displacement by the gap recovers a
    per-poll dy estimate through a single dropout; gaps beyond
    `max_gap` return None as before.
    """
    if cur_y is None or prev_y is None or not 1 <= gap_polls <= max_gap:
        return None
    return round((cur_y - prev_y) / gap_polls)


def compute_arm_centroid(
    cur_bgr: np.ndarray,
    prev_bgr: np.ndarray | None,
) -> tuple[int | None, int]:
    """Return (centroid_y, pixel_count) for the moving-white region
    between cur_bgr and prev_bgr. centroid_y is None when prev is None
    or fewer than MIN_AREA pixels moved (centroid undefined / unreliable)."""
    if prev_bgr is None or prev_bgr.shape != cur_bgr.shape:
        return None, 0

    diff = cv2.absdiff(cur_bgr, prev_bgr)
    motion_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    motion_mask = (motion_gray > MOTION_THRESHOLD).astype(np.uint8) * 255

    hsv = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, WHITE_HSV_LOW, WHITE_HSV_HIGH)

    arm_mask = cv2.bitwise_and(motion_mask, white_mask)
    coords = cv2.findNonZero(arm_mask)
    if coords is None:
        return None, 0
    area = int(len(coords))
    if area < MIN_AREA:
        return None, area
    centroid_y = int(np.mean(coords[:, 0, 1]))
    return centroid_y, area
