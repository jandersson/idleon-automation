"""Tests for the fishing landing-acceptance logic (_landing_from_detections).

The lure must SETTLE (>= MIN_LANDING_DETECTIONS sightings near the farthest x)
for a landing to count — a single mid-arc / false-match flicker is rejected so
it doesn't pollute the fill->distance fit (#58). The IO poll is separate.
"""
from minigames.fishing.main import (
    _landing_from_detections, _classify_catch, _play_crop_from_bar,
    vanished_fish, attribute_catch, _streak_after, PLAY_PAD_L, PLAY_PAD_V,
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


# --- vanished_fish: the converged multi-catch set (#66) ---------------------

def test_vanished_single_fish_caught():
    # the lone green near the landing is gone after -> caught
    assert vanished_fish([_f(100, "green")], [], landed_x=104) == [_f(100, "green")]


def test_vanished_empty_when_fish_still_there():
    # it slid in place within radius -> still present, not caught
    assert vanished_fish([_f(100, "green")], [_f(102, "green")], landed_x=104) == []


def test_vanished_converged_eel_and_squid():
    # the documented +5: eel+squid converged and both consumed
    pre = [_f(250, "eel"), _f(247, "squid")]
    caught = vanished_fish(pre, [], landed_x=249)
    assert {f["kind"] for f in caught} == {"eel", "squid"} and len(caught) == 2


def test_vanished_only_the_caught_one_when_other_slid_in_place():
    # eel caught, squid slid but stayed within radius -> only the eel vanished
    pre = [_f(250, "eel"), _f(247, "squid")]
    post = [_f(255, "squid")]
    caught = vanished_fish(pre, post, landed_x=250)
    assert [f["kind"] for f in caught] == ["eel"]


def test_vanished_ignores_fish_outside_radius():
    # a far fish isn't part of this landing's catch
    assert vanished_fish([_f(50, "green")], [], landed_x=180) == []


def test_vanished_pairs_same_kind_by_position():
    # two greens converged, one slid (to 106), one caught -> exactly one vanished
    pre = [_f(100, "green"), _f(108, "green")]
    post = [_f(106, "green")]
    assert len(vanished_fish(pre, post, landed_x=104)) == 1


def test_vanished_does_not_count_a_fish_that_slid_out_of_radius():
    # review case (1): a green slid to 122 (18px > the 15px catch radius). Kind-only
    # matching counted BOTH near greens vanished (false +2); position-aware pairing
    # pairs one to the post sighting, so only the genuinely-gone one is counted.
    pre = [_f(104, "green"), _f(100, "green")]
    post = [_f(122, "green")]
    assert len(vanished_fish(pre, post, landed_x=104)) == 1


def test_vanished_empty_when_both_near_fish_slid_away():
    # two greens near the landing both slid out of radius (a real eel, undetected,
    # was the catch): both pair to their post sightings -> nothing falsely counted
    # caught, so attribute_catch falls back to the delta's 'eel' guess (#66 review).
    pre = [_f(104, "green"), _f(100, "green")]
    post = [_f(122, "green"), _f(128, "green")]
    assert vanished_fish(pre, post, landed_x=104) == []


# --- attribute_catch: kind label + per-fish count, delta cross-checked (#66/#70)

def test_attribute_lone_eel_vs_two_greens_disambiguated_by_vanished_set():
    # a +2 delta alone is ambiguous; the vanished set names it
    assert attribute_catch([_f(0, "eel")], 2) == ("eel", 1)
    assert attribute_catch([_f(0, "green"), _f(5, "green")], 2) == ("green+green", 2)


def test_attribute_converged_eel_squid():
    assert attribute_catch([_f(0, "eel"), _f(5, "squid")], 5) == ("eel+squid", 2)


def test_attribute_label_is_sorted():
    # order-independent label so 'eel+squid' counts the same however detected
    assert attribute_catch([_f(0, "squid"), _f(5, "eel")], 5)[0] == "eel+squid"


def test_attribute_rejects_value_sum_mismatch():
    # vanished sum (5) != delta (2): a fish slid out of radius / flicker -> fall back
    assert attribute_catch([_f(0, "eel"), _f(5, "squid")], 2) == (None, None)


def test_attribute_empty_or_nonpositive_delta_falls_back():
    assert attribute_catch([], 3) == (None, None)
    assert attribute_catch([_f(0, "green")], 0) == (None, None)
    assert attribute_catch([_f(0, "green")], None) == (None, None)


def test_attribute_lone_green_confirms_count_one():
    assert attribute_catch([_f(0, "green")], 1) == ("green", 1)


def test_attribute_ignores_zero_value_fish_in_count_and_sum():
    # A non-scoring fish (megalodon = 0 pts; off in the detector today but be
    # robust) must not inflate the streak count nor make the sum match: a green
    # (+1) caught alongside a megalodon is still a 1-fish 'green' catch, not 2.
    assert attribute_catch([_f(0, "green"), _f(5, "megalodon")], 1) == ("green", 1)
    # And a lone zero-value vanish can never be a catch (delta would have to be 0).
    assert attribute_catch([_f(0, "megalodon")], 0) == (None, None)


# --- _streak_after: per-fish advance + whale reset (#70, review-flagged) -----

def test_streak_advances_by_one_for_a_lone_fish():
    assert _streak_after(5, "green", 1) == 6


def test_streak_advances_by_fish_count_for_a_converged_catch():
    assert _streak_after(5, "eel+squid", 2) == 7


def test_streak_resets_on_a_lone_whale():
    assert _streak_after(8, "whale", 1) == 1


def test_streak_resets_on_a_converged_catch_containing_a_whale():
    # 'green+whale' must still reset — the whale rule fires on any whale in the set
    assert _streak_after(8, "green+whale", 2) == 1


def test_streak_multi_label_without_whale_advances_by_count():
    # an opaque 'multi' (no attribution) advances by the fallback n_caught=1
    assert _streak_after(3, "multi", 1) == 4


def test_streak_none_label_is_safe():
    assert _streak_after(3, None, 1) == 4


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
