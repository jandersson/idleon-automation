"""Synthetic-image tests for the fishing detector — cast-bar detection and
the bar-restriction that keeps world scenery out of the colour masks (#63).

Mirrors the repo's CV-against-synthetic-images convention (no live frames).
HSV ranges are calibrated against real frames manually; these guard the
LOGIC: that the bar is found and that off-bar blobs are excluded.
"""
import cv2
import numpy as np

from minigames.fishing.detector import find_cast_bar, find_fish


def _bgra(h=200, w=700):
    """A BGRA frame (mss format) on a neutral dark background."""
    img = np.zeros((h, w, 4), np.uint8)
    img[:, :, 3] = 255
    img[:, :, :3] = (70, 70, 70)
    return img


def _fill(img, y0, y1, x0, x1, hsv):
    """Paint a rectangle a known HSV colour (set via HSV2BGR)."""
    bgr = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0, 0]
    img[y0:y1, x0:x1, :3] = [int(c) for c in bgr]


# bar: high-saturation blue (H~110 S~186 V~136); green fish: H~78.
_BAR_HSV = [110, 186, 136]
_GREEN_HSV = [78, 200, 180]


def test_find_cast_bar_locates_wide_blue_strip():
    img = _bgra()
    _fill(img, 95, 109, 100, 600, _BAR_HSV)   # wide blue track
    bar = find_cast_bar(img)
    assert bar is not None
    x, y, w, h = bar
    assert w >= 250 and w / h >= 6
    assert 90 <= x <= 110


def test_find_cast_bar_ignores_narrow_blue():
    # a small blue speck (not a wide track) must not be taken for the bar.
    img = _bgra()
    _fill(img, 95, 109, 100, 140, _BAR_HSV)
    assert find_cast_bar(img) is None


def test_bar_restriction_keeps_on_bar_fish_drops_scenery():
    img = _bgra()
    _fill(img, 95, 109, 100, 600, _BAR_HSV)      # bar
    _fill(img, 96, 108, 300, 322, _GREEN_HSV)    # green fish ON the bar
    _fill(img, 165, 185, 350, 372, _GREEN_HSV)   # green 'plant' OFF the bar
    bar = find_cast_bar(img)
    on = find_fish(img, bar=bar)
    unrestricted = find_fish(img, bar=None)
    # the on-bar fish survives the restriction...
    assert any(d["kind"] == "green" and 290 <= d["x"] <= 332 for d in on)
    # ...and the off-bar plant (y~175) does not
    assert all(d["y"] < 130 for d in on)
    # the unrestricted search catches the plant too (the false positive)
    assert len(unrestricted) > len(on)
