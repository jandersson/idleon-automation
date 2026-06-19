"""Unit tests for the fishing PTS score reader (minigames/fishing/score.py) and
the white-fill binarizer it drives (common/score_digits.py).

Synthetic only — no live captures (per CLAUDE.md). The white-fill path is what
lets the reader survive fishing's busy tan-dock background, where the shared
Otsu-minority default bridges the digits into the plank grooves (#58/#63); these
tests pin its behaviour (isolate the bright fill, reject saturated/dark scenery,
strip a banner line) and the score-delta -> catch mapping.
"""
import cv2
import numpy as np
import pytest

from common import score_digits as SD
from minigames.fishing import score as FS

_GLYPHS = SD.load_glyph_templates(FS.DIGIT_GLYPHS_DIR)


# --- kind_from_delta: the score-delta -> catch mapping ---------------------

@pytest.mark.parametrize("before,after,expected", [
    (None, 5, None),          # a read failed -> unknown, never a guess
    (3, None, None),
    (None, None, None),
    (5, 5, ("miss", 0)),      # no change -> a clean miss
    (0, 1, ("green", 1)),     # +1 green (only a lone green sums to 1)
    (4, 6, ("eel", 2)),       # +2 -> a lone eel (best guess)
    (1, 4, ("squid", 3)),     # +3 -> a lone squid (best guess)
    (0, 2, ("eel", 2)),
    # The delta is the TOTAL points: sliding fish CONVERGE and are caught
    # together, so a sum that isn't a lone common fish is a 'multi' catch worth
    # its delta (NOT a whale — +5 was observed as eel+squid).
    (2, 7, ("multi", 5)),     # +5: whale OR eel+squid -> multi (whales undetected)
    (0, 4, ("multi", 4)),     # +4: no single fish -> a converged multi-catch
    (3, 11, ("multi", 8)),    # a big jump is still a real catch worth its points
    (5, 3, None),             # a DECREASE is a bogus read -> None
])
def test_kind_from_delta(before, after, expected):
    assert FS.kind_from_delta(before, after) == expected


def test_delta_kind_is_fish_value_inverse():
    # The delta->kind map must invert the points the bot scores per kind, or a
    # caught fish would be mis-pointed. (green1/eel2/squid3/whale5.)
    from minigames.fishing.cast_model import FISH_VALUE
    for pts, kind in FS.DELTA_KIND.items():
        assert FISH_VALUE[kind] == pts


# --- white-fill binarizer ---------------------------------------------------

def _dock_crop():
    """A bright white ring 'digit' on a SATURATED tan dock, crossed by a dark
    plank groove — the case Otsu-minority bridges and white-fill must not."""
    crop = np.full((20, 60, 3), (70, 140, 200), np.uint8)  # BGR tan: high S
    crop[10:11, :] = (50, 70, 90)                          # dark plank line (V<190)
    cv2.rectangle(crop, (12, 5), (18, 15), (240, 240, 240), 1)  # white ring "0"
    return crop


def test_white_fill_isolates_bright_glyph_only():
    binary = SD.binarize_white_fill(_dock_crop())
    comps = SD.digit_components(_dock_crop(), SD.binarize_white_fill)
    # Exactly the ring survives: the saturated dock and the dark plank are gone.
    assert len(comps) == 1
    _patch, (x, _y, w, h) = comps[0]
    assert 10 <= x <= 14 and w <= 9 and h >= 8
    # The plank row is not bright+desaturated, so it never enters the mask.
    assert binary[10, 0] == 0


def test_white_fill_rejects_a_saturated_colour_blob():
    # A saturated red blob (like the charge thermometer left of the digit) must
    # NOT survive as a leading component — it's the "noise blob" that breaks the
    # read. White-fill keys on desaturated brightness, so it's excluded.
    crop = np.full((20, 60, 3), (70, 140, 200), np.uint8)
    crop[4:16, 2:8] = (40, 40, 230)            # saturated red blob, high V high S
    cv2.rectangle(crop, (24, 5), (30, 15), (240, 240, 240), 1)  # the white digit
    comps = SD.digit_components(crop, SD.binarize_white_fill)
    assert len(comps) == 1
    assert comps[0][1][0] >= 20                # leftmost component IS the digit


def test_white_fill_strips_full_width_banner_line():
    crop = np.full((20, 60, 3), (70, 140, 200), np.uint8)
    crop[2:3, :] = (240, 240, 240)             # a bright full-width banner underline
    cv2.rectangle(crop, (12, 6), (18, 16), (240, 240, 240), 1)  # the digit
    comps = SD.digit_components(crop, SD.binarize_white_fill)
    assert len(comps) == 1                     # the line is stripped, digit kept
    assert comps[0][1][3] >= 8


def test_white_fill_grayscale_fallback_thresholds_brightness():
    gray = np.full((20, 40), 80, np.uint8)     # dark bg
    cv2.rectangle(gray, (10, 5), (16, 15), 240, 1)  # bright ring
    binary = SD.binarize_white_fill(gray)
    assert binary.shape == gray.shape
    assert binary.max() == 255 and binary[0, 0] == 0


def test_white_fill_empty_crop_is_safe():
    assert SD.binarize_white_fill(np.empty((0, 0, 3), np.uint8)).size == 0


# --- the reader bound to the white-fill binarizer ---------------------------

def _ring(crop, x):
    cv2.rectangle(crop, (x, 5), (x + 6, 15), (240, 240, 240), 1)


def _bar(crop, x):
    crop[5:16, x + 2:x + 4] = (240, 240, 240)
    crop[6:8, x:x + 2] = (240, 240, 240)


def _dock(w=120):
    return np.full((20, w, 3), (70, 140, 200), np.uint8)


def test_white_fill_reader_round_trip(tmp_path):
    # Templates captured through the white-fill pipeline, then read back through
    # the same binarizer on a busy dock background.
    t0 = SD.digit_components(_then(_dock(40), _ring, 10), SD.binarize_white_fill)[0][0]
    t1 = SD.digit_components(_then(_dock(40), _bar, 10), SD.binarize_white_fill)[0][0]
    SD.save_digit_template(tmp_path, 0, t0)
    SD.save_digit_template(tmp_path, 1, t1)

    reader = SD.make_pts_reader(tmp_path, binarize=SD.binarize_white_fill)
    crop = _dock(160)
    _bar(crop, 10)
    _ring(crop, 20)
    crop[5:15, 44:66] = (240, 240, 240)        # a wide "PTS"-like blob (dropped)
    assert reader(crop) == 10
    assert reader(None) is None


def _then(crop, draw, x):
    draw(crop, x)
    return crop


# --- masked-ZNCC glyph reader: background-invariant PTS reading (#82) --------
# Compose synthetic crops from the REAL committed glyph templates (assets/
# digit_glyphs) — the glyph ink is neutral (white fill, dark outline), so it's
# pasted as gray over a COLOURED background, simulating the player-anchored
# digits over whatever scenery sits behind them. No live captures.

def _paste_digit(crop_bgr, digit, x, y=2):
    """Paint digit `digit`'s glyph ink (neutral gray) onto crop_bgr at (x, y),
    leaving the surrounding background untouched (the ink is opaque, the rest
    shows scenery — the live situation)."""
    glyph, mask = _GLYPHS[digit][0]
    h, w = glyph.shape
    roi = crop_bgr[y:y + h, x:x + w]
    sel = mask > 0
    roi[sel] = glyph[sel][:, None]        # gray ink -> equal BGR channels


def _render(bg_bgr, placements, size=(14, 60)):
    """A BGR score crop: a flat coloured background with digits placed at the
    given (digit, x) spots."""
    crop = np.zeros((size[0], size[1], 3), np.uint8)
    crop[:] = bg_bgr
    for digit, x in placements:
        _paste_digit(crop, digit, x)
    return crop


# tan dock vs pale turquoise water (the #82 area move): the two backgrounds the
# white-fill binarizer could not both handle with one threshold.
_DOCK_BGR = (70, 140, 200)
_WATER_BGR = (210, 205, 150)


def test_zncc_reader_reads_single_digit_over_any_background():
    for bg in (_DOCK_BGR, _WATER_BGR, (0, 0, 0), (255, 255, 255)):
        for d in range(10):
            crop = _render(bg, [(d, 12)])
            assert SD.read_pts_zncc(crop, _GLYPHS) == d, (d, bg)


def test_zncc_reader_is_background_invariant_same_number():
    # The core #82 property: the SAME number reads identically over the dock and
    # the pale water that broke the binarizer.
    placements = [(1, 10), (5, 18)]
    dock = SD.read_pts_zncc(_render(_DOCK_BGR, placements), _GLYPHS)
    water = SD.read_pts_zncc(_render(_WATER_BGR, placements), _GLYPHS)
    assert dock == water == 15


def test_zncc_reader_assembles_multi_digit_left_to_right():
    assert SD.read_pts_zncc(_render(_DOCK_BGR, [(2, 10), (6, 18)]), _GLYPHS) == 26
    assert SD.read_pts_zncc(_render(_WATER_BGR, [(1, 10), (9, 17)]), _GLYPHS) == 19


def test_zncc_reader_reads_many_repeats_of_one_digit():
    # A digit VALUE repeated more than GLYPH_MAX_PEAKS times must not truncate or
    # corrupt: the per-template NMS ceiling scales with crop width, so "1111111"
    # reads whole (regression for the fixed hard cap of 6).
    crop = _render(_DOCK_BGR, [(1, 6 + 6 * i) for i in range(7)], size=(14, 70))
    assert SD.read_pts_zncc(crop, _GLYPHS) == 1111111
    crop8 = _render(_WATER_BGR, [(8, 6 + 7 * i) for i in range(6)], size=(14, 70))
    assert SD.read_pts_zncc(crop8, _GLYPHS) == 888888


def test_zncc_reader_stops_at_a_label_block():
    # A solid bright block after the digits (the "PTS"/"BEST" label) is flat ->
    # low ZNCC against every digit, so it neither reads as a digit nor extends
    # the number.
    crop = _render(_WATER_BGR, [(7, 10)])
    crop[3:11, 26:48] = (245, 245, 245)        # wide solid block past the digit
    assert SD.read_pts_zncc(crop, _GLYPHS) == 7


def test_zncc_reader_returns_none_on_no_digits():
    # Flat background and random colour noise both have no glyph -> None (never
    # a fabricated 0; callers treat None as 'unknown').
    assert SD.read_pts_zncc(_render(_WATER_BGR, []), _GLYPHS) is None
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, size=(14, 60, 3), dtype=np.uint8)
    # colour noise is high-variance but not a glyph; the gate (0.7) rejects it.
    assert SD.read_pts_zncc(noise, _GLYPHS) is None
    assert SD.read_pts_zncc(None, _GLYPHS) is None


def test_zncc_reader_handles_bgra_crop():
    # The live path passes an mss BGRA crop; _to_gray must convert it.
    crop = _render(_DOCK_BGR, [(8, 12)])
    bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    assert SD.read_pts_zncc(bgra, _GLYPHS) == 8


def test_glyph_templates_load_complete_set():
    # The reader needs all 10 digits; each has a same-shape mask.
    assert set(_GLYPHS) == set(range(10))
    for d, variants in _GLYPHS.items():
        for glyph, mask in variants:
            assert glyph.shape == mask.shape and mask.max() == 255


def test_make_zncc_reader_matches_module_reader():
    reader = SD.make_zncc_pts_reader(FS.DIGIT_GLYPHS_DIR)
    crop = _render(_DOCK_BGR, [(3, 10), (4, 18)])
    assert reader(crop) == 34 == SD.read_pts_zncc(crop, _GLYPHS)
    assert reader(None) is None


def test_read_score_uses_zncc_reader_and_keeps_interface():
    # read_score(full_frame, cast_bar) crops via score_crop then reads. A None
    # anchor still yields None (unchanged contract).
    assert FS.read_score(np.zeros((50, 80, 3), np.uint8), None) is None


# --- the shared default stays unchanged (catching's path) -------------------

def test_default_binarize_is_back_compat():
    # digit_components with no binarize arg == the Otsu-minority default, so
    # catching's reader is byte-for-byte unchanged by the new parameter.
    sky = np.full((20, 60), 230, np.uint8)
    sky[5:16, 12:14] = 20
    a = SD.digit_components(sky)
    b = SD.digit_components(sky, SD._default_binarize)
    assert [box for _, box in a] == [box for _, box in b]
    assert len(a) == 1


def test_catching_reexports_white_fill():
    # Back-compat surface: callers importing from catching.score get the new
    # white-fill names too.
    from minigames.catching import score as CS
    assert CS.binarize_white_fill is SD.binarize_white_fill
    assert CS.WHITE_FILL_V_MIN == SD.WHITE_FILL_V_MIN


# --- _resolve_catch: score-delta overrides the heuristic --------------------

def test_resolve_catch_score_rescues_far_cast():
    from minigames.fishing.main import _resolve_catch
    # No bobber measured -> heuristic is (None, None, None); score +2 = eel.
    assert _resolve_catch((None, None, None), 4, 6) == ("eel", 1, "score")


def test_resolve_catch_score_overrides_landing_kind():
    from minigames.fishing.main import _resolve_catch
    # Landing guessed green; the score says +3 (squid) -> score wins.
    assert _resolve_catch(("green", 1, "landing"), 0, 3) == ("squid", 1, "score")


def test_resolve_catch_zero_delta_does_not_veto_confirmed_catch():
    from minigames.fishing.main import _resolve_catch
    # The counter can lag the catch by a frame, so a zero read right after the
    # landing may be STALE — it must NOT flip a heuristic-confirmed catch to a
    # streak-resetting miss (the dangerous failure). Keep the catch.
    assert _resolve_catch(("green", 1, "landing"), 7, 7) == ("green", 1, "landing")


def test_resolve_catch_zero_delta_confirms_an_unconfirmed_miss():
    from minigames.fishing.main import _resolve_catch
    # The heuristic also said miss -> the zero is trustworthy, attribute to score.
    assert _resolve_catch(("miss", 0, "landing"), 7, 7) == ("miss", 0, "score")


def test_resolve_catch_zero_delta_records_unmeasured_miss():
    from minigames.fishing.main import _resolve_catch
    # Far cast, no bobber measured (heuristic None) -> _measure_landing waited the
    # full settle so the zero is reliable: record the miss (resets the streak).
    assert _resolve_catch((None, None, None), 7, 7) == ("miss", 0, "score")


def test_resolve_catch_keeps_heuristic_when_a_read_is_none():
    from minigames.fishing.main import _resolve_catch
    assert _resolve_catch(("green", 1, "landing"), None, 5) == ("green", 1, "landing")
    assert _resolve_catch(("eel", 1, "eel_absence"), 3, None) == ("eel", 1, "eel_absence")


def test_resolve_catch_keeps_heuristic_on_dropped_score():
    from minigames.fishing.main import _resolve_catch
    # The score DROPPED (a bogus read) -> None -> the heuristic verdict stands.
    assert _resolve_catch(("miss", 0, "landing"), 7, 3) == ("miss", 0, "landing")


def test_resolve_catch_records_converged_multi_catch():
    from minigames.fishing.main import _resolve_catch
    # The +5 the bot once logged "whale" was eel+squid converged; +4 etc. are real
    # catches now (were dropped as ambiguous). They record as a 'multi' catch.
    assert _resolve_catch((None, None, None), 16, 21) == ("multi", 1, "score")
    assert _resolve_catch(("green", 1, "landing"), 0, 4) == ("multi", 1, "score")
