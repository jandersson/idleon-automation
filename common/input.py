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


def _charge_step(charge: int, elapsed_s: float, seen_positive: bool,
                 target_charge: int, ready_grace_s: float,
                 max_hold_s: float) -> str:
    """Closed-loop charge decision for one poll of charge_and_release. Pure.

    Returns one of:
      'release' — the charge reached target_charge, or max_hold_s elapsed: cast.
      'abort'   — the rod isn't ready (the fill stayed 0 past ready_grace_s,
                  e.g. the previous lure is still reeling in): don't cast.
      'hold'    — keep charging.

    `seen_positive` is sticky: True once any poll has read charge > 0, so a rod
    that IS charging never trips the not-ready abort on a transient 0 read."""
    if charge >= target_charge:
        return "release"
    if elapsed_s >= max_hold_s:
        return "release"
    if not seen_positive and elapsed_s >= ready_grace_s:
        return "abort"
    return "hold"


def charge_and_release(x: int, y: int, target_charge: int, read_charge,
                       poll_s: float = 0.025, ready_grace_s: float = 0.3,
                       max_hold_s: float = 1.5, jitter: int = 3) -> tuple[int, bool]:
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
      actual_charge — the bar fill at release (the model's training feature).
      ready — False when the rod wasn't ready: the fill stayed 0 through
        ready_grace_s, so the cast was aborted early. The caller should SKIP it
        (don't log it as a cast, don't count it) and retry after a beat — this
        is the previous lure still reeling in, the cast-too-soon waste (#58).

    mouseUp runs in a finally so an abort/crash never leaves the button down.
    check_failsafe() runs each poll so a corner-slam aborts mid-charge."""
    x += random.randint(-jitter, jitter)
    y += random.randint(-jitter, jitter)
    pyautogui.moveTo(x, y)
    pyautogui.mouseDown()
    charge = 0
    seen_positive = False
    start = time.time()
    try:
        while True:
            check_failsafe()
            charge = read_charge()
            seen_positive = seen_positive or charge > 0
            step = _charge_step(charge, time.time() - start, seen_positive,
                                target_charge, ready_grace_s, max_hold_s)
            if step == "release":
                return charge, True
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
