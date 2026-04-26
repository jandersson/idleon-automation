"""Tests for common/templates.py — multi-scale template matching."""
import cv2
import numpy as np

from common.templates import match_multiscale, match_multiscale_center


def _make_marker(size: tuple[int, int]) -> np.ndarray:
    """A high-contrast pseudo-random marker (deterministic seed) with strong
    internal variance — TM_CCOEFF_NORMED needs non-uniform pixels to score
    matches reliably. Plain white squares fail with zero-variance edge cases.
    """
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)


def _make_image_with_marker(image_size: tuple[int, int], marker: np.ndarray, at: tuple[int, int]):
    img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    x, y = at
    mh, mw = marker.shape[:2]
    img[y:y + mh, x:x + mw] = marker
    return img


def test_finds_marker_at_native_scale():
    marker = _make_marker((40, 40))
    image = _make_image_with_marker((400, 300), marker, at=(100, 80))

    top_left, val, scale = match_multiscale(image, marker, scales=(0.9, 1.0, 1.1))
    assert top_left is not None
    assert abs(top_left[0] - 100) <= 1
    assert abs(top_left[1] - 80) <= 1
    assert val > 0.9
    assert scale == 1.0


def test_finds_marker_at_half_size_template():
    """If the live image's marker is smaller than the captured template,
    multi-scale should find it at scale < 1."""
    template = _make_marker((40, 40))
    # Resize the marker to half size before placing in image.
    half = cv2.resize(template, (20, 20), interpolation=cv2.INTER_AREA)
    image = _make_image_with_marker((400, 300), half, at=(150, 100))

    top_left, val, scale = match_multiscale(image, template, scales=(0.4, 0.5, 0.6, 0.75, 1.0))
    assert top_left is not None
    assert scale == 0.5
    assert abs(top_left[0] - 150) <= 2
    assert abs(top_left[1] - 100) <= 2


def test_returns_none_when_template_bigger_than_image():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    template = _make_marker((40, 40))

    top_left, val, scale = match_multiscale(image, template, scales=(1.0, 1.5))
    assert top_left is None


def test_region_restricts_search():
    """Two markers — restricting the region should pick the in-region one."""
    marker = _make_marker((40, 40))
    image = _make_image_with_marker((400, 300), marker, at=(50, 50))
    image[200:240, 300:340] = marker  # second copy

    # Region covering only the second marker.
    top_left, val, scale = match_multiscale(image, marker, region=(280, 180, 400, 260), scales=(1.0,))
    assert top_left is not None
    assert top_left[0] == 300
    assert top_left[1] == 200


def test_match_multiscale_center_returns_box_center():
    marker = _make_marker((40, 40))
    image = _make_image_with_marker((400, 300), marker, at=(100, 80))

    center, val, scale = match_multiscale_center(image, marker, scales=(1.0,))
    assert center is not None
    # Center of a 40x40 box at (100,80) is (120, 100).
    assert abs(center[0] - 120) <= 1
    assert abs(center[1] - 100) <= 1
