"""Cross-pass gold wait — the pure fire/hold decision (issue #46, thread 3).

The same-sweep gold upgrade (main.py, GOLD_RIDE_MAX_MS) only defers a
green fire when gold lies AHEAD of the leaf before any red. But gold sits
somewhere on the bar for ~95% of long-session polls while only ~42% of
registered green chops had it reachable — the rest is behind the leaf or
past a red, i.e. reachable only on a LATER pass. Since chops ramp the
speed and waiting is free (docs/chopping_notes.md), the round's score is
bounded by chops, not time: converting a green (+1) chop into a gold (+2)
chop is a pure +1 point for a few seconds of wall clock.

This module holds the wait state machine so it stays unit-testable
without pyautogui / screen capture. The main loop consults it ONLY at a
would-fire moment (leaf over green, time-to-red gate passed, same-sweep
ride declined) — between those moments the wait state just persists.

Guard rails (why the wait can't cause a starve exit): a wait only STARTS
when the last fire was recent (START_LATEST_S — early/mid round, where
safe windows come every ~1-3s) and always ENDS at MAX_WAIT_S, so the
worst case adds START_LATEST_S + MAX_WAIT_S ≈ 22s since the last click —
well inside STARVE_EXIT_S=60. Layout re-rolls only happen on chops, so
gold can't genuinely vanish mid-wait; the layout_has_gold abort exists
for detection flicker.
"""

# Cap on one cross-pass wait. A full sweep under the eased model takes
# pi*W/(2*V_max) ~= 1.3-1.6s at observed speeds, so 12s is ~8 sweeps /
# several passes over any gold zone — if none of them offered a safe
# gold window, the gold is red-flanked at this speed; take greens.
MAX_WAIT_S = 12.0

# Don't start a wait when the last fire is older than this: late-round,
# safe windows are 26-45s apart and skipping one gambles the whole chop.
START_LATEST_S = 10.0


def update_gold_wait(
    wait_since: float | None,
    now: float,
    layout_has_gold: bool,
    since_last_fire_s: float,
    start_latest_s: float = START_LATEST_S,
    max_wait_s: float = MAX_WAIT_S,
) -> tuple[bool, float | None]:
    """Decide whether a safe GREEN fire should be held to wait for gold.

    Called only at a would-fire moment. Returns (hold, wait_since):
    hold=True means skip this fire and keep polling; wait_since is the
    updated wait-start timestamp (None = no wait active). The caller
    resets its wait state to None after any actual click.

    A deadline expiry returns (False, wait_since) — the caller fires the
    green and can log now - wait_since as the wait it paid for nothing.
    """
    if not layout_has_gold:
        return False, None
    if wait_since is None:
        if since_last_fire_s > start_latest_s:
            return False, None
        return True, now
    if now - wait_since >= max_wait_s:
        return False, wait_since
    return True, wait_since
