"""Mining detector.

Per the trace analysis (issue #1 resolution), the cart sprite is fixed
on screen while the minigame is active and the track scrolls past it
from right to left. So `find_cart` locates the cart per-frame (its
screen x depends on the player character's world position when the
minigame opens), and `find_next_terrain` scans the plank to the right
of the cart for incoming obstacles.

Obstacle types (verified visually against trace_20260515_220235 and
trace_20260516_131332):

- Pit: dark gap in the wooden plank with dark spikes. Lose-condition
  if the cart falls in.
- Ore: copper/tan-colored chunks protruding UP from the plank
  surface. Slamming on top rebounds the cart (free jump) and scores.

The plank's screen y-position varies by window size (the minigame UI
doesn't scale linearly), so we auto-detect the plank-top y per frame
by scanning for the brightest tan-hued horizontal band. Pit, ore, and
cart scans are all anchored to that y. This keeps the detector
window-size-independent.

Cart detection uses multi-template matching against a small set of
cart sprites captured at different window resolutions. The template
that scores highest above CART_MATCH_THRESHOLD wins. Add new templates
to assets/cart_*.png if matching fails at a new resolution.
"""
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from common.templates import match_multiscale_center

_HERE = Path(__file__).parent

# Scans start a few px past the cart's right edge so its trailing-edge
# pixels don't read as the start of an obstacle.
SCAN_BUFFER_PX = 10

# Cart matching: try every template in assets/cart_*.png at multi-scale,
# pick the best match above CART_MATCH_THRESHOLD.
CART_MATCH_THRESHOLD = 0.80
CART_SCALES = (0.5, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5)
_cart_templates: Optional[list[Tuple[str, np.ndarray]]] = None  # lazy-loaded

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
    """Locate the cart sprite. Returns (center_x, center_y) in frame
    coords, or None if no template matched above CART_MATCH_THRESHOLD.

    Search is restricted to a band around the auto-detected plank-top y;
    the cart always sits on the plank."""
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    plank_y = _find_plank_top_y(frame)
    if plank_y is None:
        return None
    return _find_cart_at_plank(frame, plank_y)


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
    # cart[0] is center_x; the cart sprite has a width that varies by
    # resolution, so estimate cart_right from the template that matched
    # via _find_cart_at_plank's cached width. As a fallback, use a small
    # multiple of plank thickness.
    cart_right = cart[0] + _estimate_cart_half_width(frame, plank_y)
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


def _load_cart_templates() -> list[Tuple[str, np.ndarray]]:
    """Lazy-load all assets/cart_*.png templates."""
    global _cart_templates
    if _cart_templates is None:
        templates = []
        for p in sorted((_HERE / "assets").glob("cart_*.png")):
            t = cv2.imread(str(p))
            if t is not None:
                templates.append((p.stem, t))
        _cart_templates = templates
    return _cart_templates


def _cart_search_region(frame, plank_y: int) -> Tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) for the cart-search ROI. The cart can be
    anywhere from the very top of the minigame overlay (peak of a jump)
    down to seated on the plank, so we let the search run from frame top
    to plank_y + 15. The minigame UI is in the upper portion of the
    window so we also cap at plank_y + 15 below to skip the score row
    and lower UI."""
    h, w = frame.shape[:2]
    return (0, 0, w, min(h, plank_y + 15))


def _find_cart_at_plank(frame, plank_y: int) -> Optional[Tuple[int, int]]:
    """Multi-template match the cart in a band above the plank. Returns
    the best match's center, or None if no template scored above
    CART_MATCH_THRESHOLD."""
    templates = _load_cart_templates()
    if not templates:
        return None
    region = _cart_search_region(frame, plank_y)
    best_center = None
    best_val = -1.0
    for _name, t in templates:
        center, val, _ = match_multiscale_center(frame, t, region=region, scales=CART_SCALES)
        if center is not None and val > best_val:
            best_val = val
            best_center = center
    if best_val < CART_MATCH_THRESHOLD:
        return None
    return best_center


def _estimate_cart_half_width(frame, plank_y: int) -> int:
    """Approximate cart half-width in pixels — used to compute cart_right
    from cart_center. Derived from the matched template's scaled width."""
    templates = _load_cart_templates()
    if not templates:
        return 30
    region = _cart_search_region(frame, plank_y)
    best_w = 0
    best_val = -1.0
    for _name, t in templates:
        center, val, scale = match_multiscale_center(frame, t, region=region, scales=CART_SCALES)
        if val > best_val:
            best_val = val
            best_w = int(round(t.shape[1] * scale))
    return max(15, best_w // 2)


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
