import pyautogui
import random
import time

pyautogui.FAILSAFE = True


def click(x: int, y: int, jitter: int = 3):
    x += random.randint(-jitter, jitter)
    y += random.randint(-jitter, jitter)
    pyautogui.click(x, y)


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
