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

from minigames.catching.main import fly_started_moving, START_MOTION_RANGE_PX


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
