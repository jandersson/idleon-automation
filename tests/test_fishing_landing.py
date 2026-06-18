"""Tests for the fishing landing-acceptance logic (_landing_from_detections).

The lure must SETTLE (>= MIN_LANDING_DETECTIONS sightings near the farthest x)
for a landing to count — a single mid-arc / false-match flicker is rejected so
it doesn't pollute the fill->distance fit (#58). The IO poll is separate.
"""
from minigames.fishing.main import (
    _landing_from_detections, _classify_catch, _play_crop_from_bar,
    PLAY_PAD_L, PLAY_PAD_V,
)


def _f(x, kind="green"):
    return {"x": x, "kind": kind}


# --- _play_crop_from_bar: the dynamic, player-position-independent crop --------
# A fixed crop clipped the bar's left edge off-screen when the player moved,
# mis-anchoring the charge/score/origin and breaking a whole session (#58). The
# crop now follows the detected bar; the load-bearing invariant is that
# reconstructing full-window coords from (crop, bar-in-crop) recovers the TRUE bar.

def test_play_crop_anchors_to_the_bar():
    play, bar = _play_crop_from_bar((434, 233, 304, 7), win_w=960, win_h=572)
    assert bar == (PLAY_PAD_L, PLAY_PAD_V, 304, 7)          # bar sits inside the crop, not clipped
    assert (play["left"] + bar[0], play["top"] + bar[1]) == (434, 233)


def test_play_crop_clamps_at_window_edges_keeping_invariant():
    # Bar near the top-left corner: the crop can't go negative, so the bar isn't
    # fully padded — but the reconstruction invariant must still hold.
    play, bar = _play_crop_from_bar((10, 5, 300, 7), win_w=960, win_h=572)
    assert play["left"] == 0 and play["top"] == 0
    assert (play["left"] + bar[0], play["top"] + bar[1]) == (10, 5)


def test_play_crop_invariant_holds_anywhere():
    for bx, by in [(0, 0), (434, 233), (200, 233), (700, 500), (650, 560)]:
        play, bar = _play_crop_from_bar((bx, by, 300, 7), 960, 572)
        assert play["left"] + bar[0] == bx and play["top"] + bar[1] == by
        assert play["width"] > 0 and play["height"] > 0


def test_catch_when_fish_near_landing_vanishes():
    # run-14 cast 6: green@109 before, gone after the lure lands at 107 -> catch
    pre = [_f(109, "green"), _f(60, "eel")]
    post = [_f(59, "eel")]
    assert _classify_catch(pre, post, landed_x=107) == ("green", 1)


def test_miss_when_fish_still_there():
    # the lure landed next to a fish but it's still sitting there -> not caught
    pre = [_f(100, "green")]
    post = [_f(100, "green")]
    assert _classify_catch(pre, post, landed_x=104) == ("miss", 0)


def test_miss_when_lure_overshoots_the_fish():
    # run-14 cast 3: fish at 203, lure landed at 226 (23px past) -> miss
    pre = [_f(203, "green")]
    post = [_f(203, "green")]
    assert _classify_catch(pre, post, landed_x=226) == ("miss", 0)


def test_miss_on_empty_water():
    assert _classify_catch([_f(50, "green")], [], landed_x=180) == ("miss", 0)


def test_catch_reports_nearest_fish_kind():
    pre = [_f(120, "eel"), _f(108, "green")]   # green is nearer the landing
    post = []
    assert _classify_catch(pre, post, landed_x=110) == ("green", 1)


def _d(x, y=40):
    return (x, y, object())   # frame placeholder


def test_single_flicker_rejected():
    assert _landing_from_detections([_d(93)]) is None       # run-13 cast 2 case


def test_empty_rejected():
    assert _landing_from_detections([]) is None


def test_settled_landing_accepted_at_farthest():
    # lure sat at ~113 for several polls -> trusted, returns the farthest
    dets = [_d(113), _d(113), _d(112), _d(113)]
    res = _landing_from_detections(dets)
    assert res is not None and res[0] == 113


def test_landing_then_reel_returns_the_max():
    # lands at 209 (twice), then reels in (lower x) -> landing is the max 209
    dets = [_d(209), _d(209), _d(188), _d(150)]
    res = _landing_from_detections(dets)
    assert res[0] == 209


def test_two_far_apart_flickers_rejected():
    # two unrelated single sightings far apart: the max (150) has only itself
    # within stable_px -> not a settle -> rejected
    assert _landing_from_detections([_d(90), _d(150)]) is None


def test_min_dets_near_max_required():
    # 3 sightings but only one is near the farthest (200); the cluster at ~120
    # doesn't confirm the 200 -> rejected
    assert _landing_from_detections([_d(118), _d(120), _d(200)]) is None
