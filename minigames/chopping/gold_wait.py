"""Gold chase — the pure hold/fire decision for gold-seeking (issue #46).

Two mechanisms defer a safe GREEN fire toward a +2 gold chop:

- the same-sweep RIDE (main.py GOLD_RIDE_MAX_MS): gold ahead of the leaf
  before any red — ride to it this sweep;
- the cross-pass WAIT (2026-07-06): gold elsewhere on the bar — hold
  green fires and let a later pass reach it. Chops ramp the speed and
  waiting is free (docs/chopping_notes.md), so a round's score is
  bounded by chops, not time: green (+1) → gold (+2) is a pure +1 point
  for a few seconds of wall clock. In the 2026-06-11 data gold sat on
  the bar in ~95% of long-session polls while 20/48 registered green
  chops fired with it unreachable same-sweep.

Both share ONE chase clock, and that's the load-bearing lesson from the
first live run (2026-07-06 18:32, stalled at chop 9): the zone layout
only re-rolls on a registered chop, so between chops the gold's
position — and whether it's red-flanked past the time-to-red gate — is
FIXED. That session's post-chop-9 layout had gold ~29ms from red;
every safe green pass had the gold "ahead within GOLD_RIDE_MAX_MS", the
deadline-less ride held fire on all of them, and the bot deferred
greens for ~20s until the user failsafed. A gold that hasn't offered a
safe window after several sweeps never will at this speed: give up and
take greens until the next re-roll makes a fresh layout.

Guard rails vs the starve exit: a chase only STARTS cross-pass when the
last fire was recent (START_LATEST_S; rides may start any time — their
nominal cost is sub-second) and every chase ENDS at MAX_CHASE_S, so the
worst case adds START_LATEST_S + MAX_CHASE_S ≈ 16s since the last
click — well inside STARVE_EXIT_S=60.
"""

# Give-up deadline for one gold chase (ride + wait combined). A full
# bar sweep under the eased model takes pi*W/(2*V_max) ~= 0.9-1.6s at
# observed speeds, so 6s is ~4 sweeps — several passes over any gold
# zone. If none offered a safe gold window, the gold is red-flanked at
# this speed and the layout won't change until the next chop.
MAX_CHASE_S = 6.0

# Don't start a cross-pass wait when the last fire is older than this:
# late-round, safe windows are 26-45s apart and skipping one gambles
# the whole chop.
START_LATEST_S = 10.0


def update_gold_chase(
    chase_since: float | None,
    now: float,
    layout_has_gold: bool,
    since_last_fire_s: float,
    same_sweep: bool,
    start_latest_s: float = START_LATEST_S,
    max_chase_s: float = MAX_CHASE_S,
) -> tuple[bool, float | None, bool]:
    """Decide whether a safe GREEN fire should be held to chase gold.

    Called only at a would-fire moment (time-to-red gate already
    passed), by the ride path (same_sweep=True, gold ahead this sweep)
    or the wait path (same_sweep=False, gold elsewhere on the bar).

    Returns (hold, chase_since, gave_up): hold=True means skip this
    fire and keep polling. gave_up=True means the chase hit its
    deadline — the caller stops chasing THIS layout's gold entirely
    (both paths) until the next re-roll, and fires the green now.
    The caller resets its chase state after any actual click.
    """
    if not layout_has_gold:
        return False, None, False
    if chase_since is None:
        if not same_sweep and since_last_fire_s > start_latest_s:
            return False, None, False
        return True, now, False
    if now - chase_since >= max_chase_s:
        return False, chase_since, True
    return True, chase_since, False
