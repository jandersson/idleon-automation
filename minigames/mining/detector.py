"""Mining detector.

Per the trace analysis (issue #1 resolution), the cart sprite is fixed
on screen and the track scrolls past it from right to left. So
`find_cart` returns a constant position and `find_next_terrain` scans
the plank to the right of the cart for incoming obstacles.

Obstacle types (verified visually against trace_20260515_220235 and
trace_20260516_131332):

- Pit: dark gap in the wooden plank with dark spikes. Lose-condition
  if the cart falls in.
- Ore: copper/tan-colored chunks protruding UP from the plank
  surface. Slamming on top rebounds the cart (free jump) and scores.

The plank's screen y-position varies by window size (the minigame UI
doesn't scale linearly), so we auto-detect the plank-top y per frame
by scanning for the brightest tan-hued horizontal band. Pit and ore
scans are then anchored to that y. This keeps the detector
window-size-independent.
"""
from typing import Optional, Tuple

import cv2
import numpy as np

# Cart sprite — currently best-effort. The cart's x is roughly fixed at
# the left side of the visible plank; we use a fraction-of-width default
# until #3 puts an actual cart picker in regions.json. Scans start a
# couple px past the cart's right edge so its trailing-edge pixels don't
# read as the start of an obstacle.
CART_X_FRAC = 0.34
CART_RIGHT_FRAC = 0.40
SCAN_BUFFER_PX = 10

# Plank-surface signature (HSV): tan/wood — H in [5,20], S>=80, V>=120.
PLANK_H_LO, PLANK_H_HI = 5, 20
PLANK_S_MIN = 80
PLANK_V_MIN = 120
PLANK_MIN_SCORE = 60  # plank-signature pixel count required to claim detection

# Plank visible x range as fraction of window width — outside this is
# cave wall, not actual track.
PLANK_X0_FRAC, PLANK_X1_FRAC = 0.20, 0.75

# Pit scan: rows just below the plank top, in the plank's body. Dark
# columns (V<PIT_V_MAX) for at least PIT_MIN_WIDTH px == pit.
PIT_SCAN_DY = (2, 7)  # rows below plank top
PIT_V_MAX = 80
PIT_MIN_WIDTH = 5

# Ore scan: rows ABOVE the plank top (chunks protrude up into normally
# dark cave-wall area). Bright tan-ish columns there == ore.
ORE_SCAN_DY = (-15, -2)  # rows above plank top
ORE_V_MIN = 80
ORE_MIN_WIDTH = 5


def find_cart(frame) -> Optional[Tuple[int, int]]:
    """Return cart sprite center (x, y). The cart's screen position is
    fixed while the minigame is active; we anchor y to the detected
    plank-top so window-size doesn't matter."""
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    plank_y = _find_plank_top_y(frame)
    if plank_y is None:
        return None
    w = frame.shape[1]
    return (int(CART_X_FRAC * w), plank_y - 6)


def find_next_terrain(frame, cart) -> Optional[dict]:
    """Return the nearest obstacle (pit or ore) ahead of the cart.

    Result: {"kind": "pit"|"ore", "x": int, "distance_px": int} where
    x is the left edge of the obstacle and distance_px is x - cart_right.
    """
    if cart is None:
        return None
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    plank_y = _find_plank_top_y(frame)
    if plank_y is None:
        return None
    w = frame.shape[1]
    cart_right = int(CART_RIGHT_FRAC * w)
    scan_x = cart_right + SCAN_BUFFER_PX
    pits = _scan_plank_pits(frame, plank_y, x_start=scan_x)
    ores = _scan_plank_ore(frame, plank_y, x_start=scan_x)
    candidates = (
        [("pit", a, b) for a, b in pits] +
        [("ore", a, b) for a, b in ores]
    )
    if not candidates:
        return None
    nearest = min(candidates, key=lambda c: c[1])
    return {"kind": nearest[0], "x": nearest[1], "distance_px": nearest[1] - cart_right}


def _find_plank_top_y(frame) -> Optional[int]:
    """Locate the brightest tan-hued horizontal band — the plank top.

    The minigame overlay sits at roughly 20-45% of window height in both
    1032- and 572-tall traces, so restrict the search to that window
    rather than the whole frame. Requires at least PLANK_MIN_SCORE
    plank-signature pixels in the best row to claim a detection — this
    keeps overworld tan-textured surfaces from being mistaken for the
    minigame plank when the minigame isn't active."""
    h, w = frame.shape[:2]
    x0 = int(PLANK_X0_FRAC * w)
    x1 = int(PLANK_X1_FRAC * w)
    y_min = int(0.20 * h)
    y_max = int(0.45 * h)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    band = hsv[y_min:y_max, x0:x1]
    is_plank = ((band[:, :, 0] >= PLANK_H_LO) & (band[:, :, 0] <= PLANK_H_HI) &
                (band[:, :, 1] >= PLANK_S_MIN) & (band[:, :, 2] >= PLANK_V_MIN))
    score_per_row = is_plank.sum(axis=1)
    best_y = int(np.argmax(score_per_row))
    if score_per_row[best_y] < PLANK_MIN_SCORE:
        return None
    return best_y + y_min


def _scan_plank_pits(frame, plank_y: int, x_start: Optional[int] = None) -> list[Tuple[int, int]]:
    """Return list of (x_left, x_right) for pits on the plank.

    x_start (default = PLANK_X0_FRAC*w) lets the caller skip past the
    cart, since the cart itself is dark relative to the plank and would
    otherwise read as a giant pit."""
    h, w = frame.shape[:2]
    x0 = x_start if x_start is not None else int(PLANK_X0_FRAC * w)
    x1 = int(PLANK_X1_FRAC * w)
    y0 = plank_y + PIT_SCAN_DY[0]
    y1 = plank_y + PIT_SCAN_DY[1]
    if y0 < 0 or y1 > h or x1 <= x0:
        return []
    band = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    is_pit = hsv[:, :, 2] < PIT_V_MAX
    col_is_pit = is_pit.sum(axis=0) > (y1 - y0) // 2
    return _extract_runs(col_is_pit, offset=x0, min_width=PIT_MIN_WIDTH)


def _scan_plank_ore(frame, plank_y: int, x_start: Optional[int] = None) -> list[Tuple[int, int]]:
    """Return list of (x_left, x_right) for ore deposits ABOVE the plank.

    Same x_start convention as the pit scanner — caller passes
    cart_right_x to skip the cart's own protrusion above the plank."""
    h, w = frame.shape[:2]
    x0 = x_start if x_start is not None else int(PLANK_X0_FRAC * w)
    x1 = int(PLANK_X1_FRAC * w)
    y0 = plank_y + ORE_SCAN_DY[0]
    y1 = plank_y + ORE_SCAN_DY[1]
    if y0 < 0 or y1 > h or x1 <= x0:
        return []
    band = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    # Tan-ish bright pixels above plank — same tan hue as plank itself
    # but in a y-band that's normally cave-dark.
    is_ore = ((hsv[:, :, 0] >= PLANK_H_LO) & (hsv[:, :, 0] <= PLANK_H_HI) &
              (hsv[:, :, 2] >= ORE_V_MIN))
    col_is_ore = is_ore.sum(axis=0) > 2
    return _extract_runs(col_is_ore, offset=x0, min_width=ORE_MIN_WIDTH)


def _extract_runs(col_mask, offset: int, min_width: int) -> list[Tuple[int, int]]:
    runs: list[Tuple[int, int]] = []
    in_run = False
    start = 0
    for x, p in enumerate(col_mask):
        if p and not in_run:
            in_run, start = True, x
        elif not p and in_run:
            runs.append((start + offset, x + offset))
            in_run = False
    if in_run:
        runs.append((start + offset, len(col_mask) + offset))
    return [(a, b) for (a, b) in runs if (b - a) >= min_width]
