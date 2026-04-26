"""Tests for chopping detector — leftmost-column and analyze_bar zone lookup."""
import numpy as np

from minigames.chopping.detector import _leftmost_column, analyze_bar


def test_leftmost_column_finds_first_qualifying_column():
    mask = np.zeros((20, 100), dtype=np.uint8)
    # Place a "leaf" with 5 vertical pixels at x=30..32.
    mask[5:10, 30:33] = 255
    # And isolated noise at x=10 (only 1 pixel).
    mask[5, 10] = 255

    # min_pixels_per_col = 2 should skip the noise and find x=30.
    assert _leftmost_column(mask, min_pixels_per_col=2) == 30


def test_leftmost_column_returns_none_on_empty_mask():
    mask = np.zeros((20, 100), dtype=np.uint8)
    assert _leftmost_column(mask, min_pixels_per_col=2) is None


def test_leftmost_column_returns_none_when_all_below_threshold():
    mask = np.zeros((20, 100), dtype=np.uint8)
    # Every column has 1 pixel — below the default 2-pixel threshold.
    mask[0, :] = 255
    assert _leftmost_column(mask, min_pixels_per_col=2) is None


def _bar_with_zones(green=(20, 60), gold=(60, 80), red_low=(0, 20), red_high=(80, 100)):
    """Make a synthetic 100x20 bar with colored zones in BGRA."""
    bgra = np.zeros((20, 100, 4), dtype=np.uint8)
    bgra[..., 3] = 255
    # BGR colors: green=(0,255,0), gold/yellow=(0,200,255), red=(0,0,255)
    bgra[:, green[0]:green[1], 1] = 255  # green
    bgra[:, gold[0]:gold[1], 1] = 200
    bgra[:, gold[0]:gold[1], 2] = 255  # gold ≈ yellow
    bgra[:, red_low[0]:red_low[1], 2] = 255  # red
    bgra[:, red_high[0]:red_high[1], 2] = 255  # red
    return bgra


def _leaf_at(x_start: int, width: int = 4):
    """A 30-tall leaf strip; bright green leaf body in BGRA."""
    leaf = np.zeros((30, 100, 4), dtype=np.uint8)
    leaf[..., 3] = 255
    # Leaf is bright green (matching LEAF_HSV: H 30-80, S/V 60+)
    leaf[5:25, x_start:x_start + width, 1] = 220
    return leaf


def test_analyze_bar_returns_green_for_leaf_over_green_zone():
    bar = _bar_with_zones()
    leaf = _leaf_at(x_start=40)  # 40 is inside the green zone (20..60)

    leaf_x, zone = analyze_bar(bar, leaf_frame=leaf)
    assert leaf_x == 40
    assert zone == "green"


def test_analyze_bar_returns_gold_for_leaf_over_gold_zone():
    bar = _bar_with_zones()
    leaf = _leaf_at(x_start=65)  # gold zone is 60..80

    _, zone = analyze_bar(bar, leaf_frame=leaf)
    assert zone == "gold"


def test_analyze_bar_returns_red_for_leaf_over_red_zone():
    bar = _bar_with_zones()
    leaf = _leaf_at(x_start=5)  # red_low is 0..20

    _, zone = analyze_bar(bar, leaf_frame=leaf)
    assert zone == "red"


def test_analyze_bar_returns_none_when_leaf_absent():
    bar = _bar_with_zones()
    leaf = np.zeros((30, 100, 4), dtype=np.uint8)
    leaf[..., 3] = 255  # all-black leaf strip — no leaf present

    leaf_x, zone = analyze_bar(bar, leaf_frame=leaf)
    assert leaf_x is None
    assert zone == "none"
