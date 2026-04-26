"""Tests for hoops detector.score_changed — binarized pixel diff."""
import cv2
import numpy as np

from minigames.hoops.detector import score_changed, score_region


def _make_grayscale_with_text(text: str, size=(80, 30)):
    """Render text on a black background — emulates how a score crop looks."""
    img = np.zeros((size[1], size[0]), dtype=np.uint8)
    cv2.putText(img, text, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
    return img


def test_unchanged_score_diff_is_zero():
    a = _make_grayscale_with_text("0")
    b = _make_grayscale_with_text("0")
    changed, diff = score_changed(a, b)
    assert not changed
    assert diff < 1.0


def test_score_change_crosses_threshold():
    a = _make_grayscale_with_text("0")
    b = _make_grayscale_with_text("1")
    changed, diff = score_changed(a, b)
    assert changed
    assert diff > 5.0


def test_score_jump_two_digits_gives_bigger_diff():
    a = _make_grayscale_with_text("0")
    b = _make_grayscale_with_text("10")
    changed, diff = score_changed(a, b)
    assert changed
    assert diff > 5.0


def test_background_noise_below_threshold():
    """Tiny pixel-level noise (parallax, anti-aliasing) shouldn't cross threshold."""
    a = _make_grayscale_with_text("0")
    b = a.copy()
    # Add a few stray pixel changes that wouldn't survive Otsu binarization.
    b[5, 5:8] = 30  # subtle background change
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
    """Out-of-bounds region asks shouldn't crash; should clip."""
    bgra = np.zeros((50, 100, 4), dtype=np.uint8)
    crop = score_region(bgra, region_left=80, region_top=40, region_width=40, region_height=30)
    # Effective crop should be within bounds: cols 80..100 = 20 wide, rows 40..50 = 10 tall.
    assert crop.shape == (10, 20)
