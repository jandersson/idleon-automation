import pyautogui
import random
import time

import mss

pyautogui.FAILSAFE = True

# pyautogui's default FAILSAFE_POINTS is just [(0, 0)] — only the top-left
# of the primary monitor triggers. Extend to all four corners of every
# attached monitor so "slam to any corner" works on multi-monitor setups.
def _all_monitor_corners() -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    with mss.mss() as sct:
        # sct.monitors[0] is the combined virtual screen; 1+ are individual.
        for mon in sct.monitors[1:]:
            l, t, w, h = mon["left"], mon["top"], mon["width"], mon["height"]
            points.extend([
                (l, t),
                (l + w - 1, t),
                (l, t + h - 1),
                (l + w - 1, t + h - 1),
            ])
    return points or [(0, 0)]


pyautogui.FAILSAFE_POINTS = _all_monitor_corners()


def click(x: int, y: int, jitter: int = 3):
    x += random.randint(-jitter, jitter)
    y += random.randint(-jitter, jitter)
    pyautogui.click(x, y)


def hold(x: int, y: int, duration_ms: int, jitter: int = 3):
    """Press and hold the left mouse button at (x, y) for duration_ms, then
    release. The fishing minigame's cast is a hold-to-charge mechanic: the
    hold duration sets the cast distance (release to cast). Position gets
    the same +/-jitter as click(); per the "Idleon clicks are buttons"
    finding, only the hold timing should matter, not where it lands.

    mouseUp runs in a finally so a crash/abort mid-hold never leaves the
    button stuck down."""
    x += random.randint(-jitter, jitter)
    y += random.randint(-jitter, jitter)
    pyautogui.moveTo(x, y)
    pyautogui.mouseDown()
    try:
        time.sleep(max(0, duration_ms) / 1000.0)
    finally:
        pyautogui.mouseUp()


# Release-lead (the "lead cast"): the fill keeps rising during the ~mouseUp+poll
# latency between the release DECISION and the click landing, so releasing exactly
# when charge >= target overshoots — and the faster the bar rises, the worse it is
# (measured: overshoot correlates with rise rate, r=0.58, implied latency ~0.076s;
# the far eel and explore casts blew past target by 6-57 charge). So release when
# the fill is PROJECTED to reach target by the time the click lands:
# charge + rise_rate * CHARGE_RELEASE_LEAD_S. The lead scales with the live rate,
# so slow casts (the common, already-accurate case) barely move while fast casts
# get a big lead — it can't overcorrect a slow cast. CHARGE_RELEASE_LEAD_S is set
# BELOW the measured latency so the residual is a slight overshoot, never an
# undershoot (a short cast misses too, and undershoot would be a NEW failure).
CHARGE_RELEASE_LEAD_S = 0.05    # release this many seconds of rise before target
CHARGE_RELEASE_LEAD_MAX = 12    # cap the predicted lead (charge px) — a noisy rate
                                # spike can't trigger a wildly early release
CHARGE_RATE_SAMPLES = 4         # rise rate is measured over the last N polls
                                # (smooths the per-poll jitter of the coarse fill)


def _release_lead(charge_rate: float,
                  release_lead_s: float = CHARGE_RELEASE_LEAD_S) -> float:
    """Charge (px) the fill is predicted to rise during the release latency:
    ``rate * lead_s``, clamped non-negative and capped. Used both to DECIDE the
    release and to ESTIMATE the landed cast power (reading + this lead) — see
    charge_and_release's return."""
    return min(max(0.0, charge_rate) * release_lead_s, CHARGE_RELEASE_LEAD_MAX)


def _charge_step(charge: int, elapsed_s: float, seen_charging: bool,
                 target_charge: int, ready_grace_s: float,
                 max_hold_s: float, charge_rate: float = 0.0,
                 release_lead_s: float = CHARGE_RELEASE_LEAD_S) -> str:
    """Closed-loop charge decision for one poll of charge_and_release. Pure.

    Returns one of:
      'release' — the charge is PROJECTED to reach target_charge by the time the
                  release lands (charge + rise rate * lead), or max_hold_s
                  elapsed: cast.
      'abort'   — the rod isn't ready (never started charging within
                  ready_grace_s, e.g. the previous lure is still reeling in):
                  don't cast.
      'hold'    — keep charging.

    `charge_rate` is the measured fill rise (charge px / second); the release is
    led by ``charge_rate * release_lead_s`` (capped, never negative) to cancel the
    poll+click latency that else overshoots. ``charge_rate=0`` (the default)
    reproduces the old release-at-target behaviour exactly.

    `seen_charging` is sticky: True once the fill has crossed the ready floor
    (set by the caller, not a lone red pixel), so a rod that IS charging never
    trips the not-ready abort on a transient low/zero read."""
    if charge + _release_lead(charge_rate, release_lead_s) >= target_charge:
        return "release"
    if elapsed_s >= max_hold_s:
        return "release"
    if not seen_charging and elapsed_s >= ready_grace_s:
        return "abort"
    return "hold"


def charge_and_release(x: int, y: int, target_charge: int, read_charge,
                       poll_s: float = 0.025, ready_grace_s: float = 0.3,
                       max_hold_s: float = 1.5, ready_floor: int = 2,
                       jitter: int = 3) -> tuple[int, bool]:
    """Closed-loop cast for the fishing minigame: press and hold LMB at (x, y),
    poll read_charge() each ~poll_s while the button is down, and RELEASE when
    the charge-bar fill reaches target_charge (or max_hold_s elapses).

    The cast distance is set by the charge LEVEL, and the bar fill is a clean,
    in-crop, reproducible signal (unlike an open-loop hold, where hold->charge
    drifts and the off-crop bobber landing is noisy). Releasing at a target fill
    aims robustly without depending on the hold-duration calibration.

    read_charge() -> int: grabs the play region and returns the charge-bar fill
    height (detector.find_charge_level). Called repeatedly with the button held.

    Returns (actual_charge, ready):
      actual_charge — the PROJECTED cast power at release: the fill reading plus
        the release-lead (the rise still to come during the click latency), which
        estimates the fill when the cast actually fires. This is the model's
        training feature, so it tracks what set the distance, not the pre-latency
        reading the lead pulls below target.
      ready — False when the rod wasn't ready: the fill never reached
        ready_floor through ready_grace_s, so the cast was aborted early. The
        caller should SKIP it (don't log it as a cast, don't count it) and retry
        after a beat — this is the previous lure still reeling in, the
        cast-too-soon waste (#58).

    ready_floor: the fill must cross this (px) before the rod counts as
    charging. A floor of >1 means a single stray red pixel-row in the narrow
    x<10 charge strip — the reeling bobber is itself a red object, or animation
    residue — can't latch 'charging' and suppress the not-ready abort. A real
    charging rod blows past it within a poll or two, well inside the grace.

    mouseUp runs in a finally so an abort/crash never leaves the button down.
    check_failsafe() runs each poll so a corner-slam aborts mid-charge."""
    x += random.randint(-jitter, jitter)
    y += random.randint(-jitter, jitter)
    pyautogui.moveTo(x, y)
    pyautogui.mouseDown()
    charge = 0
    seen_charging = False
    samples: list[tuple[float, int]] = []   # recent (t, charge) for the rise rate
    start = time.time()
    try:
        while True:
            check_failsafe()
            now = time.time()
            charge = read_charge()
            seen_charging = seen_charging or charge >= ready_floor
            samples.append((now, charge))
            del samples[:-CHARGE_RATE_SAMPLES]   # keep the last N polls
            rate = 0.0
            if len(samples) >= 2:
                dt = samples[-1][0] - samples[0][0]
                if dt > 0:
                    rate = max(0.0, (samples[-1][1] - samples[0][1]) / dt)
            step = _charge_step(charge, now - start, seen_charging,
                                target_charge, ready_grace_s, max_hold_s,
                                charge_rate=rate)
            if step == "release":
                # Return the PROJECTED cast power (reading + the release-lead),
                # not the bare reading: the fill keeps rising through the click
                # latency, so projected ~= the fill when the cast actually fires.
                # This keeps the charge_level -> distance model consistent — the
                # bare reading (which the lead pulls ~lead below target) would
                # double-count the lead and bias the model toward undershoot.
                return int(round(charge + _release_lead(rate))), True
            if step == "abort":
                return 0, False
            time.sleep(poll_s)
    finally:
        pyautogui.mouseUp()


def press_key(key: str):
    pyautogui.press(key)


def random_delay(min_ms: int = 80, max_ms: int = 200):
    time.sleep(random.randint(min_ms, max_ms) / 1000)


FAILSAFE_TOLERANCE = 5  # px — Windows snap-to-corner doesn't always
                        # land exactly on (0,0); multi-monitor / DPI scaling
                        # can report (1,0), (-1,0), etc., and strict equality
                        # against FAILSAFE_POINTS would never match.


def check_failsafe():
    """Abort if the mouse is near a fail-safe point (pyautogui.FAILSAFE_POINTS).

    pyautogui's built-in fail-safe only fires at the start of pyautogui calls
    (click, moveTo, etc.) — bots that spend most of their time in mss.grab +
    sleep loops won't notice a corner-snap until the next click, which can be
    several seconds later. Call this from each main loop iteration so the
    abort is responsive.
    """
    if not pyautogui.FAILSAFE:
        return
    x, y = pyautogui.position()
    for fx, fy in pyautogui.FAILSAFE_POINTS:
        if abs(x - fx) <= FAILSAFE_TOLERANCE and abs(y - fy) <= FAILSAFE_TOLERANCE:
            raise pyautogui.FailSafeException(
                f"PyAutoGUI fail-safe triggered (mouse at ({x},{y}))."
            )
