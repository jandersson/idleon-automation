"""Parse the darts wind panel into numeric (speed, angle, wind_x, wind_y).

The panel (located by the wind region in regions.json) shows "Wind:" plus
either "None" or "<N> mph" in blue text and a blue arrow sprite whose
orientation is the wind direction (full 2D — observed angles span the
right semicircle and the verticals).

Parsing (#41 step 2):
- speed: digit-template OCR (the score reader's machinery) on the
  bottom-left sub-crop, which holds "<N> mph" — the leading matched
  digits are the speed; the unmatched trailing components are 'mph'.
- direction: the arrow lives in the right half. Its chevron base corners
  are the pixels FARTHEST from the blue blob's centroid (the wide base
  out-distances the tip), so the pointing direction is the centroid→
  farthest-cluster vector flipped 180°. Validated against hand-readable
  archived states (clean RIGHT and straight DOWN both agree); accuracy
  ~±10°, i.e. sub-mph error on the components.
- encoding: wind_x = speed·cosθ, wind_y = speed·sinθ in SCREEN
  convention (y positive = down, matching platform_vy and everything
  else in the repo). Speed is recoverable as the vector norm. Calm
  ("None", no arrow pixels) → (0, None, 0.0, 0.0).
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.score_template_ocr import extract_digit_components, load_templates, match_digit

_TEMPLATES_DIR = Path(__file__).parent / "assets" / "digit_templates"
_templates = None

# HSV bounds for the arrow / speed text. The game colors the wind UI by
# tier: blue at the speeds seen through 9 mph, pink from 10 mph (first
# observed 2026-06-10, session 20:44 — the pink arrow read as calm under
# a blue-only mask and a 21-streak died to an unlogged 10 mph headwind).
# Mask is the union of all observed tier colors.
_WIND_COLOR_RANGES = [
    ((85, 80, 120), (115, 255, 255)),   # blue (<= 9 mph)
    ((130, 50, 150), (165, 255, 255)),  # pink (>= 10 mph; measured H 137-155)
]
_MIN_ARROW_PX = 50


def _wind_color_mask(hsv: np.ndarray) -> np.ndarray:
    mask = None
    for lo, hi in _WIND_COLOR_RANGES:
        m = cv2.inRange(hsv, lo, hi)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return mask


def parse_wind(panel_bgr: np.ndarray | None):
    """Parse a wind-panel crop. Returns (speed_mph, angle_deg, wind_x,
    wind_y):
      calm            -> (0, None, 0.0, 0.0)
      parsed          -> (int, float, float, float)
      unparseable     -> (None, None, None, None)  (crop missing/garbled)
    angle_deg is screen-convention: 0 = blowing right, +90 = down.
    """
    global _templates
    if panel_bgr is None or panel_bgr.size == 0:
        return None, None, None, None
    h, w = panel_bgr.shape[:2]

    # Arrow presence / direction from the right half.
    arrow = panel_bgr[:, int(w * 0.52):]
    hsv = cv2.cvtColor(arrow, cv2.COLOR_BGR2HSV)
    colored = _wind_color_mask(hsv)
    ys, xs = np.nonzero(colored)
    if len(xs) < _MIN_ARROW_PX:
        return 0, None, 0.0, 0.0  # "Wind: None"

    # Speed from the bottom-left sub-crop.
    if _templates is None:
        _templates = load_templates(_TEMPLATES_DIR)
    speed_crop = panel_bgr[int(h * 0.42):, :int(w * 0.52)]
    gray = cv2.cvtColor(speed_crop, cv2.COLOR_BGR2GRAY)
    digits = []
    for patch, _bbox in extract_digit_components(gray):
        d = match_digit(patch, _templates)
        if d is None:
            break  # first non-digit = the 'mph' letters
        digits.append(d)
    if not digits:
        return None, None, None, None
    speed = int("".join(str(d) for d in digits))

    # Direction. Geometry: centroid -> farthest-pixel cluster is the
    # chevron BASE (the wide end out-distances the tip), so the sprite's
    # tip points the opposite way. Empirically (backfill validation,
    # n=247 correct-pass throws) the wind BLOWS along the base direction,
    # i.e. opposite the tip: with that sign both physics checks agree —
    # wind_y vs arc height responds the right way, and the tip-rightward
    # states behave as the headwinds that cut hit rate 96% -> 82%. (Most
    # likely the sprite shows the wind's source, meteorological-style;
    # either way the features below are the empirically-validated FLOW
    # direction, which is what the model needs.)
    cx, cy = xs.mean(), ys.mean()
    d2 = (xs - cx) ** 2 + (ys - cy) ** 2
    idx = np.argsort(d2)[-max(8, len(xs) // 20):]
    ang = math.degrees(math.atan2(ys[idx].mean() - cy, xs[idx].mean() - cx))
    rad = math.radians(ang)
    return speed, ang, speed * math.cos(rad), speed * math.sin(rad)
