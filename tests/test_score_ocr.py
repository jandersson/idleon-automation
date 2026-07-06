"""Tests for common.score_ocr.

Skip the actual-OCR cases when the tesseract binary isn't installed
(CI machines, fresh dev setups). The interface contract — "returns int
or None, never raises" — is the part we always check.
"""
import shutil

import cv2
import numpy as np
import pytest

from common.score_ocr import read_score, _find_tesseract_binary


# The bot resolves the binary via _find_tesseract_binary (PATH plus the
# standard install dirs — winget doesn't always update PATH), so gate
# the live-OCR tests on the same lookup, not bare PATH.
HAS_TESSERACT = _find_tesseract_binary() is not None
needs_tesseract = pytest.mark.skipif(
    not HAS_TESSERACT, reason="tesseract binary not on PATH"
)


def test_returns_none_for_empty_input():
    assert read_score(np.zeros((0, 0), dtype=np.uint8)) is None


def test_returns_none_for_too_small_input():
    assert read_score(np.zeros((2, 2), dtype=np.uint8)) is None


def test_returns_none_for_uniform_input():
    """A blank/uniform crop has no digits to read."""
    crop = np.full((20, 30), 128, dtype=np.uint8)
    # Either tesseract returns empty (None) or it isn't installed (also None).
    assert read_score(crop) is None


@needs_tesseract
def test_reads_synthetic_digit():
    """Render a digit on a clean black background and verify OCR reads it.

    Uses cv2.putText with FONT_HERSHEY_SIMPLEX since it's reliably available.
    """
    crop = np.zeros((40, 30), dtype=np.uint8)
    cv2.putText(crop, "5", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
    assert read_score(crop) == 5


@needs_tesseract
def test_reads_two_digit_number():
    crop = np.zeros((40, 50), dtype=np.uint8)
    cv2.putText(crop, "23", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
    assert read_score(crop) == 23


def test_parse_leading_int():
    from common.score_ocr import parse_leading_int
    assert parse_leading_int("21 PTS") == 21
    assert parse_leading_int("  7 PTS\n") == 7
    assert parse_leading_int("PTS") is None
    assert parse_leading_int("") is None
    assert parse_leading_int(None) is None


def test_read_leading_int_returns_none_for_empty_input():
    from common.score_ocr import read_leading_int
    assert read_leading_int(np.zeros((0, 0), dtype=np.uint8)) is None
    assert read_leading_int(np.full((20, 60), 128, dtype=np.uint8)) is None


@needs_tesseract
def test_read_leading_int_ignores_text_suffix():
    """The chopping PTS counter renders '21 PTS' — the digits-only
    whitelist mangled the suffix into digits ('16 PTS' -> 1675, 21:17
    run). The leading-int reader must return just the number."""
    from common.score_ocr import read_leading_int
    crop = np.zeros((40, 160), dtype=np.uint8)
    cv2.putText(crop, "21 PTS", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
    assert read_leading_int(crop) == 21
