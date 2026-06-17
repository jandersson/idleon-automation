"""Tests for the fishing landing-acceptance logic (_landing_from_detections).

The lure must SETTLE (>= MIN_LANDING_DETECTIONS sightings near the farthest x)
for a landing to count — a single mid-arc / false-match flicker is rejected so
it doesn't pollute the fill->distance fit (#58). The IO poll is separate.
"""
from minigames.fishing.main import _landing_from_detections


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
