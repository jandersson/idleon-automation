import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import click, random_delay
from common.window import get_bounds, WindowNotFoundError
from minigames.chopping.detector import analyze_bar

WINDOW_TITLE = "Idleon"

# TODO: coordinates relative to the game window's top-left.
BAR_REGION_REL = {"left": 0, "top": 0, "width": 400, "height": 40}
CHOP_BUTTON_REL = (0, 0)

POLL_INTERVAL = 0.01
COOLDOWN_AFTER_CLICK = 0.25


def run():
    print(f"Chopping bot starting — tracking window {WINDOW_TITLE!r}. Move mouse to a corner to abort.")
    time.sleep(2)

    while True:
        try:
            win_left, win_top, _, _ = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError as e:
            print(e)
            time.sleep(1)
            continue

        frame = grab_region(
            win_left + BAR_REGION_REL["left"],
            win_top + BAR_REGION_REL["top"],
            BAR_REGION_REL["width"],
            BAR_REGION_REL["height"],
        )
        pointer_x, zone = analyze_bar(frame)

        if pointer_x is not None and zone in ("green", "gold"):
            print(f"Pointer at x={pointer_x} in {zone} zone — chopping")
            random_delay(20, 60)
            click(win_left + CHOP_BUTTON_REL[0], win_top + CHOP_BUTTON_REL[1])
            time.sleep(COOLDOWN_AFTER_CLICK)
            continue

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
