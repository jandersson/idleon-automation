"""Regression tests for catching's start-confirmation gate.

The bot was entering a new world region whose dark enemy sprites fall in the
sky-band play crop and false-trigger find_fly as a STATIC blob (a session
logged 42 flaps all at fly_y=104, vy=0, and never clicked PLAY GAME). The
fix makes the PLAY GAME button authoritative at startup and only hands
control to the fly once it's confirmed MOVING — that confirmation is
`fly_started_moving`, guarded here so a static false-positive can never be
read as a started game again.
"""
from collections import deque

from minigames.catching.main import (
    fly_started_moving,
    timed_flap_due,
    START_MOTION_RANGE_PX,
    _anchored_play_region,
    ANCHOR_PLAY_HALF_W,
    ANCHOR_PLAY_TOP_DY,
    ANCHOR_PLAY_HEIGHT,
)


def test_empty_or_single_detection_is_not_moving():
    assert fly_started_moving(deque()) is False
    assert fly_started_moving(deque([104])) is False


def test_static_blob_is_not_moving():
    # The desert-region failure: the same y every frame.
    assert fly_started_moving(deque([104, 104, 104, 104])) is False


def test_detection_jitter_is_not_moving():
    # A couple px of anti-aliasing wobble must not count as a real fly.
    assert fly_started_moving(deque([104, 106, 105, 103])) is False


def test_bobbing_or_falling_fly_is_moving():
    # A real fly free-falls / bobs over tens of px.
    assert fly_started_moving(deque([104, 120])) is True
    assert fly_started_moving(deque([90, 100, 130, 88])) is True


def test_threshold_is_inclusive():
    assert fly_started_moving(deque([100, 100 + START_MOTION_RANGE_PX])) is True
    assert fly_started_moving(deque([100, 100 + START_MOTION_RANGE_PX - 1])) is False


# --- prompt-anchored play region (#60) -------------------------------------

def test_anchored_region_centred_on_prompt():
    # The minigame band is at a fixed offset from the PLAY GAME prompt, so the
    # crop centres on the prompt x and offsets down from the prompt y.
    r = _anchored_play_region(670, 226, 959, 572)
    assert r == {"left": 670 - ANCHOR_PLAY_HALF_W,
                 "top": 226 + ANCHOR_PLAY_TOP_DY,
                 "width": 2 * ANCHOR_PLAY_HALF_W,
                 "height": ANCHOR_PLAY_HEIGHT}
    # avatar (the real (666,170) in that run) lands inside the crop
    assert r["left"] <= 666 <= r["left"] + r["width"]
    assert r["top"] <= 170 <= r["top"] + r["height"]


def test_anchored_region_clamps_to_window():
    # Prompt near the top-left: region clamps without going negative.
    r = _anchored_play_region(100, 50, 959, 572)
    assert r["left"] == 0
    assert r["top"] == 0
    assert r["left"] + r["width"] <= 959
    assert r["top"] + r["height"] <= 572


def test_anchored_region_clamps_bottom_and_right():
    r = _anchored_play_region(950, 560, 959, 572)
    assert r["left"] + r["width"] <= 959
    assert r["top"] + r["height"] <= 572
    assert r["width"] > 0 and r["height"] > 0


# --- timed_flap_due (phase-timing, #60) -------------------------------------

def test_timed_flap_due_fires_when_near_and_low():
    # leading edge 30px ahead (within LEAD_DIST), avatar below the hoop centre.
    assert timed_flap_due(fly_x=256, fly_y=100, gap_left_x=286, gap_center_y=80) is True


def test_timed_flap_not_due_when_hoop_far():
    assert timed_flap_due(fly_x=256, fly_y=100, gap_left_x=356, gap_center_y=80) is False


def test_timed_flap_not_due_when_avatar_high():
    # avatar above the centre (smaller y) — a flap there over-lifts; wait.
    assert timed_flap_due(fly_x=256, fly_y=60, gap_left_x=286, gap_center_y=80) is False


def test_timed_flap_not_due_after_leading_edge_passes():
    assert timed_flap_due(fly_x=256, fly_y=100, gap_left_x=250, gap_center_y=80) is False


def test_timed_flap_due_none_gap():
    assert timed_flap_due(256, 100, None, None) is False


def test_timed_flap_not_due_at_or_just_below_centre():
    # firing right at the centre over-lifts; require well below it (the run-12
    # crash fired at fly_y == centre).
    assert timed_flap_due(fly_x=256, fly_y=80, gap_left_x=286, gap_center_y=80) is False
    assert timed_flap_due(fly_x=256, fly_y=85, gap_left_x=286, gap_center_y=80) is False  # +5 < +6
    assert timed_flap_due(fly_x=256, fly_y=88, gap_left_x=286, gap_center_y=80) is True   # +8 >= +6
