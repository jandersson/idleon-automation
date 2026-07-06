"""Gold chase state machine (minigames/chopping/gold_wait.py).

The main loop consults update_gold_chase only at would-fire moments —
from the same-sweep ride path (same_sweep=True) or the cross-pass wait
path (same_sweep=False). These tests walk the transitions that matter:
start, hold, the shared give-up deadline (the fix for the 2026-07-06
chop-9 stall, where a red-flanked gold made the deadline-less ride hold
every safe green for ~20s), starved-round refusal, gold-gone abort.
"""
from minigames.chopping.gold_wait import update_gold_chase


def test_wait_starts_when_gold_on_bar_and_round_healthy():
    hold, since, gave_up = update_gold_chase(
        None, 100.0, True, since_last_fire_s=2.0, same_sweep=False)
    assert (hold, since, gave_up) == (True, 100.0, False)


def test_no_wait_without_gold_on_bar():
    hold, since, gave_up = update_gold_chase(
        None, 100.0, False, since_last_fire_s=2.0, same_sweep=False)
    assert (hold, since, gave_up) == (False, None, False)


def test_no_new_wait_when_round_is_starving():
    # Last fire 30s ago: safe windows are rare, don't gamble one.
    hold, since, gave_up = update_gold_chase(
        None, 100.0, True, since_last_fire_s=30.0, same_sweep=False)
    assert (hold, since, gave_up) == (False, None, False)


def test_ride_may_start_even_late_round():
    # Same-sweep rides cost sub-second nominally; the starving-round
    # gate only applies to cross-pass waits.
    hold, since, gave_up = update_gold_chase(
        None, 100.0, True, since_last_fire_s=30.0, same_sweep=True)
    assert (hold, since, gave_up) == (True, 100.0, False)


def test_holds_within_deadline():
    hold, since, gave_up = update_gold_chase(
        100.0, 103.0, True, since_last_fire_s=7.0, same_sweep=False)
    assert (hold, since, gave_up) == (True, 100.0, False)  # start preserved


def test_deadline_gives_up_and_releases_the_green():
    hold, since, gave_up = update_gold_chase(
        100.0, 107.0, True, since_last_fire_s=9.0, same_sweep=False,
        max_chase_s=6.0)
    assert hold is False
    assert gave_up is True
    # chase_since survives so the caller can log the chase duration.
    assert since == 100.0


def test_ride_shares_the_same_deadline():
    # The chop-9 stall: gold permanently "ahead within the ride cap"
    # but red-flanked. The ride path must hit the same give-up.
    hold, since, gave_up = update_gold_chase(
        100.0, 107.0, True, since_last_fire_s=9.0, same_sweep=True,
        max_chase_s=6.0)
    assert hold is False
    assert gave_up is True


def test_gold_vanishing_aborts_an_active_chase():
    hold, since, gave_up = update_gold_chase(
        100.0, 103.0, False, since_last_fire_s=7.0, same_sweep=False)
    assert (hold, since, gave_up) == (False, None, False)


def test_active_wait_ignores_start_latest_gate():
    # The starving-round gate only blocks STARTING a wait; one already
    # running keeps holding until its own deadline.
    hold, since, gave_up = update_gold_chase(
        100.0, 102.0, True, since_last_fire_s=11.0, same_sweep=False,
        start_latest_s=10.0)
    assert (hold, since, gave_up) == (True, 100.0, False)


def test_worst_case_stays_inside_starve_exit():
    # Guard-rail arithmetic: a cross-pass chase can start at most
    # START_LATEST_S after the last fire and gives up by MAX_CHASE_S,
    # so the deadline green is taken well inside the 60s starve exit.
    from minigames.chopping.gold_wait import MAX_CHASE_S, START_LATEST_S
    assert START_LATEST_S + MAX_CHASE_S < 60.0 / 2
