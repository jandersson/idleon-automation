"""Synthetic-image tests for the fishing detector — cast-bar detection and
the bar-restriction that keeps world scenery out of the colour masks (#63).

Mirrors the repo's CV-against-synthetic-images convention (no live frames).
HSV ranges are calibrated against real frames manually; these guard the
LOGIC: that the bar is found and that off-bar blobs are excluded.
"""
import cv2
import numpy as np

from pathlib import Path

from minigames.fishing.detector import (
    find_cast_bar, find_fish, find_charge_level, find_charge_fill,
    find_mines, _in_a_mine, find_eel, find_fish_sprites, ASSETS,
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


def test_find_cast_bar_accepts_right_clipped_bar():
    # Corner cut-off (game bug): the bar is clipped at the window's RIGHT edge, so
    # it's narrower than BAR_MIN_WIDTH (250) but pinned to the edge — accept it at
    # the reduced floor so the visible left portion is playable (#corner-cutoff).
    img = _bgra(h=200, w=700)
    _fill(img, 95, 103, 470, 700, _BAR_HSV)   # 230px bar, right edge == window width
    bar = find_cast_bar(img)
    assert bar is not None
    x, y, w, h = bar
    assert x == 470 and x + w == 700        # pinned to the right edge, visible width kept


def test_find_cast_bar_rejects_left_clipped_short_bar():
    # A short bar clipped at the LEFT edge cuts off the origin/charge/score anchor —
    # reject it (needs the full BAR_MIN_WIDTH), so the bot doesn't mis-anchor (#58).
    img = _bgra(h=200, w=700)
    _fill(img, 95, 103, 0, 230, _BAR_HSV)     # 230px bar pinned to the LEFT edge
    assert find_cast_bar(img) is None


def test_find_cast_bar_rejects_midscreen_narrow_bar():
    # A 230px strip NOT touching either edge is below BAR_MIN_WIDTH -> rejected
    # (only an edge-clipped bar earns the reduced floor).
    img = _bgra(h=200, w=700)
    _fill(img, 95, 103, 300, 530, _BAR_HSV)
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


def test_find_fish_keeps_green_on_a_mine():
    # A green fish sitting inside a mine bbox is a real catchable fish (landing
    # on a fish over a mine still scores), so it is kept, not excluded.
    img = _bgra()
    _fill(img, 96, 112, 296, 312, _GREEN_HSV)
    mines = [{"x": 304, "y": 104, "bbox": (292, 92, 28, 24)}]
    assert any(d["kind"] == "green" for d in find_fish(img, mines=mines))


def test_find_eel_matches_its_sprite_and_ignores_green():
    # The eel migrated to the masked-ZNCC sprite path (#77): find_eel is a wrapper
    # over find_fish_sprites, so it needs the cast bar. Pasting the real eel sprite
    # on the bar is matched; a flat green block is not (different pixels).
    img = _bgra(h=200, w=700)
    _fill(img, 95, 109, 100, 600, _BAR_HSV)
    _paste_sprite(img, "fish_eel_1.png", 300, 86)
    res = find_eel(img, bar=find_cast_bar(img))
    assert res is not None and 300 <= res[0] <= 330
    green_only = _bgra(h=200, w=700)
    _fill(green_only, 95, 109, 100, 600, _BAR_HSV)
    _fill(green_only, 95, 111, 300, 316, _GREEN_HSV)
    assert find_eel(green_only, bar=find_cast_bar(green_only)) is None


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


def test_green_bound_excludes_turquoise_water_keeps_seafoam_fish():
    # A bright-beach biome puts turquoise WATER at H~90-92, which the old green
    # upper bound (92) let flood the bar window and bury the fish -> find_fish
    # returned nothing and the bot sat stuck (#72). The bound is now 88: a
    # seafoam-green fish (H<=88) is still detected, but a turquoise square (H91)
    # on the same bar is NOT classified as a fish. Guards against widening it back.
    img = _bgra()
    _fill(img, 95, 109, 100, 600, _BAR_HSV)         # bar
    _fill(img, 96, 112, 300, 317, [85, 200, 180])   # seafoam fish (H85) ON the bar
    _fill(img, 96, 112, 380, 397, [91, 120, 200])   # turquoise water square (H91)
    bar = find_cast_bar(img)
    fish = [d for d in find_fish(img, bar=bar) if d["kind"] == "green"]
    assert any(292 <= d["x"] <= 325 for d in fish)          # the fish is found
    assert all(not (372 <= d["x"] <= 405) for d in fish)    # the H91 water is not


# --- masked-ZNCC sprite detection (#75): background-invariant fish detection ---

def _paste_sprite(img, name, x, y):
    """Paste the full real sprite crop (incl. its own corner background)."""
    s = cv2.imread(str(ASSETS / name))
    img[y:y + s.shape[0], x:x + s.shape[1], :3] = s
    return s.shape[:2]


def _composite_sprite(img, name, x, y):
    """Overlay only the MASKED sprite body, leaving img's background in the
    corners — simulates the same fish over a different world."""
    s = cv2.imread(str(ASSETS / name))
    m = cv2.imread(str(ASSETS / name.replace(".png", "_mask.png")), cv2.IMREAD_GRAYSCALE)
    roi = img[y:y + s.shape[0], x:x + s.shape[1], :3]
    roi[m > 0] = s[m > 0]


def test_find_fish_sprites_detects_real_green_template():
    img = _bgra(h=200, w=700)
    _fill(img, 95, 109, 100, 600, _BAR_HSV)
    _paste_sprite(img, "fish_green_1.png", 300, 88)   # real sprite on the bar
    sp = find_fish_sprites(img, find_cast_bar(img))
    assert any(d["kind"] == "green" and 300 <= d["x"] <= 330 for d in sp)
    assert all("conf" in d for d in sp)


def test_find_fish_sprites_is_background_invariant():
    # The SAME fish body over two very different worlds must both match (#75) —
    # absolute HSV breaks per biome (#63/#72); masked ZNCC scores only the sprite
    # pixels, so the world behind it is irrelevant.
    for bg_hsv in ([15, 180, 210], [95, 170, 200]):   # orange sunset vs turquoise water
        img = _bgra(h=200, w=700)
        bgr = cv2.cvtColor(np.uint8([[bg_hsv]]), cv2.COLOR_HSV2BGR)[0, 0]
        img[75:130, 100:600, :3] = [int(c) for c in bgr]    # the biome band
        _fill(img, 95, 109, 100, 600, _BAR_HSV)
        _composite_sprite(img, "fish_green_1.png", 300, 88)  # body only; corners = biome
        sp = find_fish_sprites(img, find_cast_bar(img))
        assert any(d["kind"] == "green" for d in sp), f"missed over bg {bg_hsv}"


def test_find_fish_sprites_rejects_a_flat_colour_blob():
    # A flat green rectangle (which the HSV gate calls a fish) is NOT the textured
    # sprite, so the ZNCC matcher rejects it — sprite detection is more specific.
    img = _bgra(h=200, w=700)
    _fill(img, 95, 109, 100, 600, _BAR_HSV)
    _fill(img, 96, 108, 300, 320, _GREEN_HSV)            # flat green block
    assert find_fish_sprites(img, find_cast_bar(img)) == []


def test_find_fish_sprites_none_bar_is_empty():
    assert find_fish_sprites(_bgra(), None) == []


def test_find_fish_merges_sprite_detection_additively():
    # find_fish surfaces the sprite detection (with its conf) alongside HSV.
    img = _bgra(h=200, w=700)
    _fill(img, 95, 109, 100, 600, _BAR_HSV)
    _paste_sprite(img, "fish_squid_1.png", 300, 86)
    fish = find_fish(img, bar=find_cast_bar(img))
    assert any(f["kind"] == "squid" and "conf" in f for f in fish)


def test_merge_keeps_converged_whale_next_to_a_green_sprite():
    # A whale (5pts, HSV-only — no sprite template) converged within DEDUP_PX of a
    # green sprite det must SURVIVE: the kind-aware dedup only suppresses a same-kind
    # HSV duplicate, so a green sprite can't delete the high-value whale (#75 review).
    img = _bgra(h=200, w=700)
    _fill(img, 95, 109, 100, 600, _BAR_HSV)
    _paste_sprite(img, "fish_green_1.png", 300, 88)          # green sprite ~x311
    whale_hsv = [109, 90, 200]                               # in FISH_HSV['whale'] (H100-118, low S)
    _fill(img, 96, 110, 314, 330, whale_hsv)                 # whale blob ~6px from the green
    fish = find_fish(img, bar=find_cast_bar(img))
    assert any(f["kind"] == "green" for f in fish)           # sprite green kept
    assert any(f["kind"] == "whale" for f in fish)           # converged whale NOT dropped


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
