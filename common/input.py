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


def check_failsafe():
    """Abort if the mouse is in a screen corner (pyautogui.FAILSAFE_POINTS).

    pyautogui's built-in fail-safe only fires at the start of pyautogui calls
    (click, moveTo, etc.) — bots that spend most of their time in mss.grab +
    sleep loops won't notice a corner-snap until the next click, which can be
    several seconds later. Call this from each main loop iteration so the
    abort is responsive.
    """
    if not pyautogui.FAILSAFE:
        return
    if pyautogui.position() in pyautogui.FAILSAFE_POINTS:
        raise pyautogui.FailSafeException(
            "PyAutoGUI fail-safe triggered from mouse moving to a corner."
        )
