"""Tests for the minigame cooldown formula (#22).

Asserts the high-confidence SHAPE/cap, not the provisional exact
constants (which need a clean-capture day to pin — see
docs/hoops_cooldown.md), so the test won't break if the constants are
later refined within the same shape.
"""
from common.idleon_save import (
    minigame_cooldown_ticks_for_plays as cd_ticks,
    predict_hoops_cooldown_seconds,
)

CAP_TICKS = 900


def test_no_cooldown_before_first_play():
    assert cd_ticks(0) == 0
    assert cd_ticks(-3) == 0


def test_first_play_is_a_small_cooldown():
    # "Starts with a small cooldown" — well under the cap.
    assert 0 < cd_ticks(1) < CAP_TICKS // 3


def test_monotonic_non_decreasing_then_capped():
    vals = [cd_ticks(n) for n in range(0, 21)]
    assert all(b >= a for a, b in zip(vals, vals[1:])), vals
    assert max(vals) <= CAP_TICKS


def test_reaches_cap_by_play_seven():
    # Quadratic escalation hits the 900-tick cap around the 7th play.
    assert cd_ticks(7) == CAP_TICKS
    assert cd_ticks(20) == CAP_TICKS


def test_seconds_wrapper_is_ticks_over_five_and_caps_at_180():
    assert predict_hoops_cooldown_seconds(3) == cd_ticks(3) / 5
    assert predict_hoops_cooldown_seconds(50) == 180.0  # 900 ticks / 5
