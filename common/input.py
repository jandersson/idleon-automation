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
