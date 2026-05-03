"""Tests for common.score_ocr.

Skip the actual-OCR cases when the tesseract binary isn't installed
(CI machines, fresh dev setups). The interface contract — "returns int
or None, never raises" — is the part we always check.
"""
import shutil

import cv2
import numpy as np
import pytest

from common.score_ocr import read_score


HAS_TESSERACT = shutil.which("tesseract") is not None
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
