import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import click, random_delay
from common.window import get_bounds, WindowNotFoundError
from minigames.darts.detector import find_target

WINDOW_TITLE = "Idleon"
POLL_INTERVAL = 0.02

# TODO: fill in once we know the dartboard's screen geometry.
# All coords are window-relative (offsets from the Idleon window's top-left),
# resolved against the live window position via get_bounds() each frame.
BOARD_REGION_REL = {"left": 0, "top": 0, "width": 400, "height": 400}
THROW_BUTTON_REL = (0, 0)

# Wait after throwing so the next dart can load.
POST_THROW_COOLDOWN = 1.0


def run():
    print(f"Darts bot starting — tracking window {WINDOW_TITLE!r}. Move mouse to a corner to abort.")
    time.sleep(2)

    while True:
        try:
            win_left, win_top, _, _ = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError as e:
            print(e)
            time.sleep(1)
            continue

        frame = grab_region(
            win_left + BOARD_REGION_REL["left"],
            win_top + BOARD_REGION_REL["top"],
            BOARD_REGION_REL["width"],
            BOARD_REGION_REL["height"],
        )

        # TODO: replace with real fire condition once mechanics are known.
        target = find_target(frame)
        if target is None:
            time.sleep(POLL_INTERVAL)
            continue

        print(f"Target at {target} — throwing")
        random_delay(20, 60)
        click(win_left + THROW_BUTTON_REL[0], win_top + THROW_BUTTON_REL[1])
        time.sleep(POST_THROW_COOLDOWN)


if __name__ == "__main__":
    run()
