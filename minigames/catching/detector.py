"""Catching minigame detectors.

The catching minigame is a Flappy-Bird-style game (wiki: "pass through the
small golden hoop in the Catching minigame"): a small dark FLY bobs at a
fixed x while golden HOOPS scroll in from the right; tap to flap the fly up
so it passes through the next hoop's opening. Two things to find per frame:

- The fly (player) — its (x, y) in the play region.
- The next hoop ahead of the fly — its vertical band (the gap to thread)
  and how far ahead it is.

Grounded in the 2026-06-17 observe frames (959x572 window): the fly is a
~10x9px dark blob in the bright sky at a fixed x (the player's minigame
position), bobbing in y; the hoop is a tall-narrow saturated-orange oval in
the upper sky. The desert below is also orange-ish, so detection is
restricted to the play region (the sky band — main.py crops to it) plus a
tall-narrow aspect filter that rejects the wide sand/HUD contours.

CALIBRATION: the thresholds below are derived from those frames; verify
against a live run (fishing-style observe) and adjust if the fly/hoop are
missed. assets/fly.png is a template fallback for find_fly.
"""
from pathlib import Path

import cv2
import numpy as np

from common.templates import match_multiscale_center

ASSETS = Path(__file__).parent / "assets"

# "PLAY GAME" entry-prompt button (assets/play_button.png is the badge-free
# "PLAY GAME" text — the count badge changes per play, so it's excluded).
# Validated against the observe frames: 1.00 when the prompt is up vs 0.55
# mid-game, so 0.75 cleanly separates. The prompt is anchored above the
# player (moves with them), hence template matching, not coordinate caching.
PLAY_BUTTON_MATCH_THRESHOLD = 0.75

# Golden hoops — saturated orange/yellow. (Other hoop colours score more —
# wiki #52: orange/green/gold/lava — but the observe frames only had
# orange; colour-aware scoring is a follow-up once those are captured.)
RING_HSV_LOWER = np.array([10, 110, 130])
RING_HSV_UPPER = np.array([35, 255, 255])
RING_MIN_AREA = 200    # ignore noise (small particles, distant rings)
RING_MAX_AREA = 50000  # ignore very-large gold UI elements
# The hoop is a vertical oval (taller than wide); requiring h >= w rejects
# the wide desert-ground / HUD-bar contours that share the orange hue.
RING_MIN_ASPECT = 1.0  # min height/width

# Fly: a small dark sprite in the bright sky. Blob detection (densest small
# dark blob) is more robust than template-matching a ~10px animated sprite.
FLY_DARK_V_MAX = 70     # fly is dark; sky V is high. At 70 only the fly
                        # qualifies in the observe frames; looser (95) also
                        # caught a dark-ish top-corner UI artifact.
FLY_MIN_AREA = 20
FLY_MAX_AREA = 220
FLY_MIN_FILL = 0.35     # contourArea / bbox area — rejects sparse edge noise
FLY_TEMPLATE_THRESHOLD = 0.6


def _to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def find_fly(frame: np.ndarray) -> tuple[int, int] | None:
    """Locate the fly as the densest small dark blob in `frame` (which
    main.py has already cropped to the play region, so the dark character
    below is excluded). Falls back to template-matching assets/fly.png.
    Returns the (x, y) centre or None."""
    bgr = _to_bgr(frame)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark = (gray < FLY_DARK_V_MAX).astype(np.uint8) * 255
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: tuple[int, int] | None = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if not FLY_MIN_AREA <= area <= FLY_MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        if area / (w * h) < FLY_MIN_FILL:
            continue          # sparse/thin (edge noise, hoop outline)
        if not 0.4 <= w / h <= 2.5:
            continue          # fly is roughly square
        if area > best_area:
            best_area = area
            best = (x + w // 2, y + h // 2)
    if best is not None:
        return best
    return _find_fly_template(bgr)


def _find_fly_template(bgr: np.ndarray) -> tuple[int, int] | None:
    fly_path = ASSETS / "fly.png"
    if not fly_path.exists():
        return None
    template = cv2.imread(str(fly_path), cv2.IMREAD_COLOR)
    if template is None or bgr.shape[0] < template.shape[0] or bgr.shape[1] < template.shape[1]:
        return None
    result = cv2.matchTemplate(bgr, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < FLY_TEMPLATE_THRESHOLD:
        return None
    th, tw = template.shape[:2]
    return (max_loc[0] + tw // 2, max_loc[1] + th // 2)


def find_play_button(frame: np.ndarray) -> tuple[int, int] | None:
    """Locate the "PLAY GAME" entry button via multi-scale template match
    against assets/play_button.png. Returns its (x, y) centre (clicking
    there starts a play) or None below PLAY_BUTTON_MATCH_THRESHOLD / when
    the template is missing. Search the FULL window frame — the prompt is
    anchored to the player's world position, not a fixed region."""
    path = ASSETS / "play_button.png"
    if not path.exists():
        return None
    template = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if template is None:
        return None
    bgr = _to_bgr(frame)
    if bgr.shape[0] < template.shape[0] or bgr.shape[1] < template.shape[1]:
        return None
    center, val, _ = match_multiscale_center(bgr, template)
    if center is None or val < PLAY_BUTTON_MATCH_THRESHOLD:
        return None
    return center


def find_next_gap(
    frame: np.ndarray, fly_pos: tuple[int, int] | None
) -> tuple[int, int, int, int] | None:
    """Find the next hoop ahead of the fly; return (top_y, bottom_y,
    left_x, right_x). The vertical span is the band to thread; the
    horizontal span is how far ahead the hoop is (logged so flap timing
    can be correlated with the hoop's approach).

    HSV-masks orange hoop pixels, keeps tall-narrow components (RING_MIN_
    ASPECT — the vertical-oval hoop, not the wide desert/HUD), and picks
    the leftmost whose centre is right of the fly (the next one in scroll
    order). Assumes `frame` is the play-region crop (so the desert ground
    below is already excluded).
    """
    if fly_pos is None:
        return None
    fly_x, _fly_y = fly_pos

    bgr = _to_bgr(frame)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, RING_HSV_LOWER, RING_HSV_UPPER)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < RING_MIN_AREA or area > RING_MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h / w < RING_MIN_ASPECT:
            continue  # wide contour = desert ground / HUD bar, not a hoop
        cx = x + w // 2
        if cx <= fly_x:
            continue  # hoop already passed
        candidates.append((cx, x, y, w, h))

    if not candidates:
        return None
    candidates.sort()  # smallest cx > fly_x = next hoop ahead
    _, x, y, w, h = candidates[0]
    return (y, y + h, x, x + w)
