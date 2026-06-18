"""Fishing PTS score reader — the ground-truth catch signal.

The "N PTS" counter renders below-left of the cast bar in the FULL window (NOT
in the play crop). Read before vs after a cast, its delta says exactly what was
caught: +1 green, +2 eel, +3 squid, +5 whale, 0 a miss — fixing the cases the
bobber/disappearance heuristics miss (far casts that don't settle, occluded
overlaps). Uses the shared common.score_digits reader with fishing's own digit
templates (assets/digit_templates), grown by `fishing-capture-digits`.

The score region is anchored to the detected cast bar (like the charge
thermometer), so it follows the player-anchored overlay. Offsets measured on the
botrun_225352 full frames: cast bar at (434,233), the "0 PTS" digit at ~(404,250).
"""
from pathlib import Path

import numpy as np

from common.score_digits import binarize_white_fill, make_pts_reader

_HERE = Path(__file__).parent
DIGIT_TEMPLATES_DIR = _HERE / "assets" / "digit_templates"

# Score crop offset from the cast bar's (x, y), full-window coords. The PTS
# NUMBER sits just left of and below the bar's left end; the crop runs through
# the "PTS" label (the reader stops at it), wide enough for ~2-3 digits.
#
# DX0 starts the crop just left of the leading digit (its left edge is at
# ~bar_x-21 in the 960x572 frames), NOT at the charge thermometer: the red
# thermometer's bright highlight sits at bar_x-38..-28 and otherwise reads as a
# spurious leading component that breaks the read (the "noise blob left of 0").
# -26 clears the thermometer with a ~5px margin before the digit (#63).
SCORE_DX0 = -26
SCORE_DX1 = 45
SCORE_DY0 = 11
SCORE_DY1 = 30

# Point value -> caught fish kind, for turning a score DELTA into a catch.
# (Wiki: green 1, eel 2, squid 3, whale 5.)
DELTA_KIND = {1: "green", 2: "eel", 3: "squid", 5: "whale"}

# White-fill binarization (not the shared Otsu-minority default): the fishing
# score renders over a busy tan dock whose dark plank grooves bridge the digits
# under Otsu. Keying on the bright white glyph fill rejects the dock, the planks,
# and the coloured bar. Templates in DIGIT_TEMPLATES_DIR are captured through
# this same binarizer (fishing-capture-digits), so live components match (#63).
_read_pts = make_pts_reader(DIGIT_TEMPLATES_DIR, binarize=binarize_white_fill)


def score_crop(full_frame: np.ndarray, cast_bar_full):
    """The PTS score region from the full window, anchored to the cast bar
    ((x, y, w, h) in full-window coords). None when the anchor is missing."""
    if cast_bar_full is None:
        return None
    bx, by = cast_bar_full[0], cast_bar_full[1]
    h, w = full_frame.shape[:2]
    x0, x1 = max(0, bx + SCORE_DX0), min(w, bx + SCORE_DX1)
    y0, y1 = max(0, by + SCORE_DY0), min(h, by + SCORE_DY1)
    if x1 <= x0 or y1 <= y0:
        return None
    return full_frame[y0:y1, x0:x1]


def read_score(full_frame: np.ndarray, cast_bar_full):
    """Current PTS score (int), or None if unreadable / a digit isn't captured
    yet (callers must treat None as 'unknown', never as 0)."""
    return _read_pts(score_crop(full_frame, cast_bar_full))


def kind_from_delta(before, after):
    """The caught fish kind from a (before, after) PTS pair, or None.

    Returns (kind, points) when the score rose by a known fish value, ("miss", 0)
    when it didn't move, or None when either read failed (unknown — don't guess).
    A multi-fish jump or an unmapped delta also returns None (ambiguous)."""
    if before is None or after is None:
        return None
    delta = after - before
    if delta == 0:
        return "miss", 0
    kind = DELTA_KIND.get(delta)
    return (kind, delta) if kind is not None else None
