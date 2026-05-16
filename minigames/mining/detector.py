"""Mining detector.

Per the trace analysis (issue #1 resolution), the cart sprite is fixed
on screen and the track scrolls past it from right to left. So
`find_cart` returns a constant position and `find_next_terrain` scans
the plank to the right of the cart for incoming obstacles.

Pit detection: pits are dark gaps in the wooden plank. Sample the
plank-surface band (y=PLANK_Y0..PLANK_Y1), threshold V<PIT_V_MAX, and
look for contiguous columns where most pixels are dark.

Ore detection: TODO — no ore-containing frames captured yet.

Constants are pixel-space against the 960x1032 game window seen on the
calibration machine. If the window resizes these will need to be
expressed as fractions, same pattern as regions.json.
"""
from typing import Optional, Tuple

import cv2

# Cart sprite (fixed on-screen position when minigame is active).
# Center of the cart body. Right edge is where obstacle collision begins.
CART_CENTER = (325, 312)
CART_RIGHT_X = 360

# Plank-surface band — bright tan pixels at the top of the wooden plank.
# Below this band the plank has a shadow line; above is empty space.
PLANK_Y0, PLANK_Y1 = 317, 322

# Pit scanner: V<PIT_V_MAX = dark = pit candidate. Above this is plank.
PIT_V_MAX = 80
PIT_MIN_WIDTH = 5

# Plank visible x range — outside this is cave wall, not actual track.
PLANK_X0, PLANK_X1 = 300, 720


def find_cart(frame) -> Optional[Tuple[int, int]]:
    """Return cart sprite center (x, y). Fixed position while the minigame
    is active; caller should treat None as 'no minigame on screen'."""
    return CART_CENTER


def find_next_terrain(frame, cart) -> Optional[dict]:
    """Return the nearest obstacle ahead of the cart, or None.

    Result: {"kind": "pit", "x": int, "distance_px": int} where x is the
    left edge of the obstacle and distance_px is x - cart_right.
    Ore is not yet implemented (no training data).
    """
    if cart is None:
        return None
    pits = _scan_plank_pits(frame)
    cart_right = CART_RIGHT_X
    ahead = [(a, b) for (a, b) in pits if a > cart_right]
    if not ahead:
        return None
    nearest = min(ahead, key=lambda r: r[0])
    return {"kind": "pit", "x": nearest[0], "distance_px": nearest[0] - cart_right}


def _scan_plank_pits(frame) -> list[Tuple[int, int]]:
    """Return list of (x_left, x_right) for each pit detected on the plank.
    Pure function — given a BGR or BGRA frame, returns pit positions in
    full-frame x coordinates."""
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    band = frame[PLANK_Y0:PLANK_Y1, PLANK_X0:PLANK_X1]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    is_pit = hsv[:, :, 2] < PIT_V_MAX
    band_h = PLANK_Y1 - PLANK_Y0
    col_is_pit = is_pit.sum(axis=0) > band_h // 2
    runs: list[Tuple[int, int]] = []
    in_run = False
    start = 0
    for x, p in enumerate(col_is_pit):
        if p and not in_run:
            in_run, start = True, x
        elif not p and in_run:
            runs.append((start + PLANK_X0, x + PLANK_X0))
            in_run = False
    if in_run:
        runs.append((start + PLANK_X0, len(col_is_pit) + PLANK_X0))
    return [(a, b) for (a, b) in runs if (b - a) >= PIT_MIN_WIDTH]
