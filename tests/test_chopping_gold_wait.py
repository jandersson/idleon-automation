"""Cross-pass gold wait state machine (minigames/chopping/gold_wait.py).

The main loop consults update_gold_wait only at a would-fire moment
(safe green, same-sweep ride declined); these tests walk the state
transitions that matter — start, hold, deadline, starved-round refusal,
gold-gone abort — with explicit timestamps, no capture or CV.
"""
from minigames.chopping.gold_wait import update_gold_wait


def test_starts_wait_when_gold_on_bar_and_round_healthy():
    hold, since = update_gold_wait(None, 100.0, True, since_last_fire_s=2.0)
    assert hold is True
    assert since == 100.0


def test_no_wait_without_gold_on_bar():
    hold, since = update_gold_wait(None, 100.0, False, since_last_fire_s=2.0)
    assert hold is False
    assert since is None


def test_no_new_wait_when_round_is_starving():
    # Last fire 30s ago: safe windows are rare, don't gamble one.
    hold, since = update_gold_wait(None, 100.0, True, since_last_fire_s=30.0)
    assert hold is False
    assert since is None


def test_holds_within_deadline():
    hold, since = update_gold_wait(100.0, 105.0, True, since_last_fire_s=7.0)
    assert hold is True
    assert since == 100.0  # wait start is preserved, not refreshed


def test_deadline_expiry_releases_the_green():
    hold, since = update_gold_wait(
        100.0, 113.0, True, since_last_fire_s=15.0, max_wait_s=12.0)
    assert hold is False
    # wait_since survives so the caller can log the wasted wait duration.
    assert since == 100.0


def test_gold_vanishing_aborts_an_active_wait():
    hold, since = update_gold_wait(100.0, 105.0, False, since_last_fire_s=7.0)
    assert hold is False
    assert since is None


def test_active_wait_ignores_start_latest_gate():
    # The starving-round gate only blocks STARTING a wait; one already
    # running keeps holding until its own deadline.
    hold, since = update_gold_wait(
        100.0, 108.0, True, since_last_fire_s=11.0, start_latest_s=10.0)
    assert hold is True
    assert since == 100.0


def test_worst_case_stays_inside_starve_exit():
    # Guard-rail arithmetic: a wait can start at most START_LATEST_S
    # after the last fire and ends by MAX_WAIT_S, so the deadline green
    # is taken well inside the 60s starve exit.
    from minigames.chopping.gold_wait import MAX_WAIT_S, START_LATEST_S
    assert START_LATEST_S + MAX_WAIT_S < 60.0 / 2
