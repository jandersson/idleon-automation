"""Tests for common/score_diff.py.

Mirrors the previous tests/test_hoops_score.py — same logic, now lives in
common/ and is importable by both hoops and darts.
"""
import cv2
import numpy as np

from common.score_diff import score_changed, score_region


def _make_grayscale_with_text(text: str, size=(80, 30)):
    img = np.zeros((size[1], size[0]), dtype=np.uint8)
    cv2.putText(img, text, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    return img


def test_unchanged_score_diff_is_zero():
    a = _make_grayscale_with_text("0")
    changed, diff = score_changed(a, a.copy())
    assert not changed
    assert diff < 1.0


def test_score_change_crosses_threshold():
    a = _make_grayscale_with_text("0")
    b = _make_grayscale_with_text("1")
    changed, diff = score_changed(a, b)
    assert changed
    assert diff > 5.0


def test_background_noise_below_threshold():
    a = _make_grayscale_with_text("0")
    b = a.copy()
    b[5, 5:8] = 30
    b[20, 50:53] = 40
    changed, diff = score_changed(a, b)
    assert not changed


def test_shape_mismatch_signals_changed():
    a = np.zeros((30, 80), dtype=np.uint8)
    b = np.zeros((20, 80), dtype=np.uint8)
    changed, diff = score_changed(a, b)
    assert changed
    assert diff == 255.0


def test_score_region_clips_to_frame_bounds():
    bgra = np.zeros((50, 100, 4), dtype=np.uint8)
    crop = score_region(bgra, region_left=80, region_top=40, region_width=40, region_height=30)
    assert crop.shape == (10, 20)


def test_score_region_returns_grayscale():
    bgra = np.zeros((50, 100, 4), dtype=np.uint8)
    bgra[..., 2] = 200  # red channel
    crop = score_region(bgra, region_left=10, region_top=10, region_width=30, region_height=20)
    assert crop.ndim == 2  # grayscale, no channel dim


def test_uniform_background_doesnt_false_positive():
    """If the score region was picked over solid background, Otsu becomes
    ill-defined: tiny pixel noise can produce a huge fake diff. The function
    should detect this and return no-change."""
    # Pre: near-uniform with tiny noise (matches a real captured background)
    pre = np.full((30, 80), 25, dtype=np.uint8)
    pre[5, 10] = 21  # a few stray pixels off by a couple of values
    pre[10, 30] = 23
    # Post: completely uniform
    post = np.full((30, 80), 26, dtype=np.uint8)
    changed, diff = score_changed(pre, post)
    assert not changed
    assert diff == 0.0
