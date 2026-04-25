import time
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import click, random_delay
from common.window import get_bounds, WindowNotFoundError
from minigames.hoops.detector import find_hoop, find_platform

WINDOW_TITLE = "Idleon"
POLL_INTERVAL = 0.02

# Vertical offset added to hoop rim Y to get the ideal platform-launch Y.
# Rolls rim-vs-platform geometry AND ball-travel lead into one number.
# Positive = launch below rim, negative = above. Tune empirically.
VERTICAL_OFFSET = 0

# Accepted window around target_y (pixels) when deciding to fire.
Y_TOLERANCE = 6

# Required direction of platform motion to fire. "up", "down", or "any".
REQUIRED_DIRECTION = "up"

# Wait after clicking so the ball can travel and the hoop can reposition.
POST_SHOT_COOLDOWN = 2.0


def _direction(history: deque) -> str:
    if len(history) < 2:
        return "any"
    delta = history[-1] - history[0]
    if delta < -1:
        return "up"  # screen y decreases upward
    if delta > 1:
        return "down"
    return "flat"


def run():
    print(f"Hoops bot starting — tracking window {WINDOW_TITLE!r}. Move mouse to a corner to abort.")
    time.sleep(2)

    platform_history: deque[int] = deque(maxlen=5)
    target_y: int | None = None

    while True:
        try:
            left, top, width, height = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError as e:
            print(e)
            time.sleep(1)
            continue

        frame = grab_region(left, top, width, height)

        if target_y is None:
            hoop_pos, hoop_conf = find_hoop(frame)
            if hoop_pos is None:
                print(f"Hoop not found (best confidence={hoop_conf:.2f})")
                time.sleep(0.2)
                continue
            target_y = hoop_pos[1] + VERTICAL_OFFSET
            print(f"Hoop rim at y={hoop_pos[1]} (conf={hoop_conf:.2f}), target launch y={target_y}")

        platform_pos, platform_conf = find_platform(frame)
        if platform_pos is None:
            time.sleep(POLL_INTERVAL)
            continue

        py = platform_pos[1]
        platform_history.append(py)

        if abs(py - target_y) <= Y_TOLERANCE:
            direction = _direction(platform_history)
            if REQUIRED_DIRECTION == "any" or direction == REQUIRED_DIRECTION:
                print(f"Platform at y={py} (target={target_y}, dir={direction}) — shooting")
                random_delay(10, 40)
                click(left + width // 2, top + height // 2)
                time.sleep(POST_SHOT_COOLDOWN)
                platform_history.clear()
                target_y = None  # re-detect hoop after it repositions
                continue

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
