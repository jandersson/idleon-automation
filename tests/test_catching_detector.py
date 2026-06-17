"""Synthetic-image tests for the catching detector (fly + hoop), #47.

Mirrors the repo's CV-against-synthetic-images convention (no live frames).
The calibrated thresholds are validated against real observe frames
manually; these guard the detection LOGIC: blob-based fly, and the
tall-narrow aspect filter that isolates the hoop from the wide desert/HUD.
"""
import cv2
import numpy as np

from minigames.catching.detector import find_fly, find_next_gap, find_play_button, ASSETS


def _sky(h=183, w=400):
    """Bright BGRA 'sky' background (mss frames are BGRA)."""
    img = np.empty((h, w, 4), np.uint8)
    img[:, :, 0] = 220   # B
    img[:, :, 1] = 210   # G
    img[:, :, 2] = 190   # R
    img[:, :, 3] = 255   # A
    return img


def _dark_blob(img, cx, cy, s=6):
    img[cy - s:cy + s, cx - s:cx + s, :3] = 25  # dark, solid (the fly)


def _orange_rect(img, x, y, w, h):
    img[y:y + h, x:x + w, 0] = 10    # B
    img[y:y + h, x:x + w, 1] = 130   # G
    img[y:y + h, x:x + w, 2] = 240   # R  -> saturated orange in HSV


# ---- find_fly -------------------------------------------------------------

def test_find_fly_locates_dark_blob():
    img = _sky()
    _dark_blob(img, 250, 90, s=6)
    pos = find_fly(img)
    assert pos is not None
    assert abs(pos[0] - 250) <= 3 and abs(pos[1] - 90) <= 3


def test_find_fly_none_on_empty_sky():
    assert find_fly(_sky()) is None


def test_find_fly_ignores_sparse_edge_noise():
    # A thin (low-fill) dark streak must be rejected; the real fly chosen.
    img = _sky()
    img[5:7, 0:60, :3] = 25       # thin horizontal streak
    _dark_blob(img, 250, 90, s=6)
    pos = find_fly(img)
    assert pos is not None and abs(pos[0] - 250) <= 3


# ---- find_next_gap --------------------------------------------------------

def test_find_next_gap_returns_tall_hoop_ahead():
    img = _sky()
    _orange_rect(img, 200, 60, 20, 60)   # tall-narrow hoop, ahead of the fly
    gap = find_next_gap(img, fly_pos=(50, 100))
    assert gap is not None
    top, bottom, left, right = gap
    assert 55 <= top <= 65 and 115 <= bottom <= 125
    assert 195 <= left <= 205


def test_find_next_gap_rejects_wide_ground_contour():
    # A wide orange band (desert ground / HUD bar) shares the hue but is
    # wider than tall -> rejected by the aspect filter.
    img = _sky()
    _orange_rect(img, 100, 150, 250, 18)
    assert find_next_gap(img, fly_pos=(50, 100)) is None


def test_find_next_gap_ignores_hoop_behind_fly():
    img = _sky()
    _orange_rect(img, 20, 60, 20, 60)    # hoop to the LEFT of the fly
    assert find_next_gap(img, fly_pos=(200, 100)) is None


def test_find_next_gap_none_without_fly():
    img = _sky()
    _orange_rect(img, 200, 60, 20, 60)
    assert find_next_gap(img, fly_pos=None) is None


# ---- find_play_button (auto-start) ---------------------------------------

def test_find_play_button_matches_pasted_template():
    tpl = cv2.imread(str(ASSETS / "play_button.png"))  # BGR (the real template)
    assert tpl is not None, "play_button.png template missing"
    th, tw = tpl.shape[:2]
    img = np.full((300, 959, 4), 200, np.uint8)
    img[:, :, 3] = 255
    y, x = 131, 600
    img[y:y + th, x:x + tw, :3] = tpl
    pos = find_play_button(img)
    assert pos is not None
    assert abs(pos[0] - (x + tw // 2)) <= 4 and abs(pos[1] - (y + th // 2)) <= 4


def test_find_play_button_none_when_absent():
    img = np.full((300, 959, 4), 200, np.uint8)
    img[:, :, 3] = 255
    assert find_play_button(img) is None
