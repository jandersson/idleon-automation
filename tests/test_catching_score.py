"""Unit tests for the catching PTS score reader (minigames/catching/score.py).

Synthetic only — no live captures (per CLAUDE.md). The digit glyphs here are
deliberately simple, distinguishable shapes (a bar vs a ring); the point is
the pipeline (binarize -> isolate components -> tolerant match -> assemble),
not the real font, which is validated visually against observe frames.
"""
import cv2
import numpy as np
import pytest

from minigames.catching import score as S


# --- parse_pts (the pure text->int helper) ---------------------------------

@pytest.mark.parametrize("text,expected", [
    ("1 PTS", 1),
    ("0 PTS", 0),
    ("10 PTS", 10),
    ("123 PTS", 123),
    ("1PTS", 1),
    ("  7  PTS  ", 7),
    ("7 BEST", 7),       # leading int regardless of trailing label
    ("PTS", None),       # no digits
    ("", None),
    (None, None),
    ("abc", None),
])
def test_parse_pts(text, expected):
    assert S.parse_pts(text) == expected


def test_parse_pts_takes_leading_run_only():
    # The number is always leftmost; a misread label after it must not merge in.
    assert S.parse_pts("12 P7S") == 12
    assert S.parse_pts("3 BE5T") == 3


# --- synthetic glyph helpers -----------------------------------------------

SKY = 230   # light background
INK = 20    # dark glyph/outline


def _bar(crop, x):
    """Draw a 'one'-like glyph at column x: a vertical stroke plus a top-left
    flag, so the bounding box contains background (a real glyph never fills
    its whole bbox — a solid block has no variance to correlate against)."""
    crop[5:16, x + 2:x + 4] = INK   # vertical stroke
    crop[6:8, x:x + 2] = INK        # top-left flag


def _ring(crop, x):
    """Draw a 'zero'-like ring (w9, h11 outline) at column x."""
    cv2.rectangle(crop, (x, 5), (x + 8, 15), INK, 1)


def _blank(w=120, h=20):
    return np.full((h, w), SKY, np.uint8)


def _digit_template(draw):
    """Extract the single-component patch a given draw fn produces, the same
    way the live reader would — so templates and reads share a pipeline."""
    crop = _blank()
    draw(crop, 10)
    comps = S.digit_components(crop)
    assert len(comps) == 1
    return comps[0][0]


# --- digit_components / line stripping -------------------------------------

def test_strip_horizontal_lines_removes_full_width_row():
    crop = _blank()
    crop[2:3, :] = INK         # a full-width platform line
    _bar(crop, 10)             # plus a real digit
    comps = S.digit_components(crop)
    # The line is stripped, so only the bar survives (not a merged giant blob).
    assert len(comps) == 1
    _, (x, _y, w, h) = comps[0]
    assert w <= 5 and h >= 10


def test_digit_components_drops_wide_pts_blob():
    crop = _blank()
    _bar(crop, 10)
    crop[5:15, 40:62] = INK    # a wide "PTS"-like blob (w22 > DIGIT_MAX_WIDTH)
    comps = S.digit_components(crop)
    assert len(comps) == 1     # only the digit; the blob is dropped by width
    assert comps[0][1][0] == 10


# --- match_digit / read_pts_from_crop --------------------------------------

def test_reader_reads_multi_digit_and_excludes_pts():
    templates = {0: _digit_template(_ring), 1: _digit_template(_bar)}
    canon = {d: S._canon(t) for d, t in templates.items()}
    crop = _blank(w=160)
    _bar(crop, 10)             # "1"
    _ring(crop, 18)            # "0"
    crop[5:15, 40:62] = INK    # "PTS" blob (excluded)
    assert S.read_pts_from_crop(crop, canon) == 10


def test_reader_stops_at_uncaptured_digit():
    # Library has only "1"; a "0" it can't match must STOP the read, not be
    # guessed — "1" then unknown -> read "1" (leading run), never a wrong digit.
    canon = {1: S._canon(_digit_template(_bar))}
    crop = _blank(w=160)
    _bar(crop, 10)
    _ring(crop, 18)
    assert S.read_pts_from_crop(crop, canon) == 1


def test_reader_returns_none_when_first_digit_unknown():
    canon = {1: S._canon(_digit_template(_bar))}
    crop = _blank()
    _ring(crop, 10)            # only an unmatchable "0"
    assert S.read_pts_from_crop(crop, canon) is None


def test_match_digit_below_threshold_returns_none():
    canon = {1: S._canon(_digit_template(_bar))}
    # A solid block matches neither well — must be rejected, not coerced.
    block = np.full((11, 9), 255, np.uint8)
    digit, scoreval = S.match_digit(block, canon)
    assert digit is None


def test_empty_crop_reads_none():
    assert S.read_pts_from_crop(_blank(), {}) is None


# --- template persistence round-trip ---------------------------------------

def test_template_roundtrip_and_reader(tmp_path):
    S.save_digit_template(tmp_path, 0, _digit_template(_ring))
    S.save_digit_template(tmp_path, 1, _digit_template(_bar))
    loaded = S.load_digit_templates(tmp_path)
    assert set(loaded) == {0, 1}

    reader = S.make_pts_reader(tmp_path)
    crop = _blank(w=160)
    _bar(crop, 10)
    _ring(crop, 18)
    assert reader(crop) == 10
    assert reader(None) is None


def test_make_pts_reader_empty_library_reads_none(tmp_path):
    reader = S.make_pts_reader(tmp_path)   # no templates on disk
    crop = _blank()
    _bar(crop, 10)
    assert reader(crop) is None            # graceful: no false-high
