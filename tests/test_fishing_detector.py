"""Synthetic-image tests for the fishing detector — cast-bar detection and
the bar-restriction that keeps world scenery out of the colour masks (#63).

Mirrors the repo's CV-against-synthetic-images convention (no live frames).
HSV ranges are calibrated against real frames manually; these guard the
LOGIC: that the bar is found and that off-bar blobs are excluded.
"""
import cv2
import numpy as np

from minigames.fishing.detector import (
    find_cast_bar, find_fish, find_charge_level, find_charge_fill,
    find_mines, _in_a_mine,
)


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


def test_find_charge_level_measures_left_red_bar():
    # the charge bar is a thin red strip at the far-left edge; its fill height
    # is the cast-power signal (#58). Empty -> 0.
    img = _bgra()
    _fill(img, 70, 100, 0, 3, [3, 220, 230])     # red bar, bottom 30 rows, x<3
    assert 28 <= find_charge_level(img) <= 32
    assert find_charge_level(_bgra()) == 0
    # a red object further right (e.g. the beach umbrella, x>=12) must NOT count
    img2 = _bgra()
    _fill(img2, 70, 100, 20, 30, [3, 220, 230])
    assert find_charge_level(img2) == 0


def test_find_mines_detects_red_spiky_ball():
    # Mines are spiky balls keyed off their RED spikes; a ~28px red blob is a
    # mine, a ~14px one (fish-sized) is not, and pale/orange scenery isn't red.
    img = _bgra()
    _fill(img, 90, 118, 300, 328, [3, 220, 230])     # 28x28 red ball
    mines = find_mines(img)
    assert len(mines) == 1 and 300 <= mines[0]["x"] <= 330
    small = _bgra()
    _fill(small, 95, 109, 300, 314, [3, 220, 230])    # 14px — fish-sized, not a mine
    assert find_mines(small) == []
    orange = _bgra()
    _fill(orange, 90, 118, 300, 328, [20, 220, 230])  # orange (sky/core hue), not red
    assert find_mines(orange) == []


def test_in_a_mine():
    mines = [{"x": 314, "y": 104, "bbox": (300, 90, 28, 28)}]
    assert _in_a_mine(314, 104, mines)
    assert _in_a_mine(300, 90, mines)        # corner, within pad
    assert not _in_a_mine(360, 104, mines)   # well outside


def test_find_fish_drops_eel_in_mine_keeps_green():
    # A mine's orange core reads as 'eel'; it's dropped when inside a mine. A
    # GREEN sitting on a mine is a real catchable fish and is kept.
    img = _bgra()
    _fill(img, 96, 112, 296, 312, _GREEN_HSV)        # green fish ~x304
    _fill(img, 96, 112, 396, 412, [20, 200, 230])    # warm 'eel' blob ~x404
    mines = [{"x": 404, "y": 104, "bbox": (392, 92, 28, 24)}]  # mine over the eel
    kinds = {(d["kind"]) for d in find_fish(img, mines=mines)}
    xs = {d["kind"]: d["x"] for d in find_fish(img, mines=mines)}
    assert "green" in kinds                          # green kept
    assert "eel" not in kinds                        # mine-core eel dropped
    # a green that happened to be inside a mine bbox is still kept
    mines2 = [{"x": 304, "y": 104, "bbox": (292, 92, 28, 24)}]
    assert any(d["kind"] == "green" for d in find_fish(img, mines=mines2))


def test_find_charge_fill_reads_thermometer_left_of_cast_bar():
    # The live charge meter is a vertical red thermometer LEFT of the cast bar;
    # find_charge_fill reads its red-fill HEIGHT anchored to the cast bar's
    # full-window (x,y). Paint a partial red fill in the anchored window and
    # assert the height; an empty anchor reads 0.
    img = _bgra(h=300, w=900)
    bar = (434, 233, 304, 7)        # cast bar in full-window coords (as live)
    # thermometer search window is bar_x-50..-22, bar_y-64..+18 => x384..412, y169..251
    # paint a 20px-tall red fill at the bottom of that window
    _fill(img, 231, 251, 392, 404, [3, 220, 230])   # red, ~20 rows, inside the window
    h = find_charge_fill(img, bar)
    assert 18 <= h <= 22
    assert find_charge_fill(_bgra(h=300, w=900), bar) == 0     # empty -> 0
    assert find_charge_fill(img, None) == 0                    # no anchor -> 0


def test_shape_filter_rejects_thin_scenery_edge():
    # The fish is a square solid sprite; the tan-scenery / bar-edge false
    # positives are thin and wide. A square fish is kept, a thin green strip
    # (same hue) is rejected by the aspect gate (#63).
    img = _bgra()
    _fill(img, 95, 109, 100, 600, _BAR_HSV)      # bar
    _fill(img, 96, 108, 300, 320, _GREEN_HSV)    # square fish ON the bar (kept)
    _fill(img, 100, 104, 130, 260, _GREEN_HSV)   # thin wide green strip (rejected)
    bar = find_cast_bar(img)
    xs = [d["x"] for d in find_fish(img, bar=bar) if d["kind"] == "green"]
    assert any(290 <= x <= 332 for x in xs)          # square fish kept
    assert all(not (175 <= x <= 215) for x in xs)    # thin strip (~x195) dropped


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
