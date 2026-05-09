"""Mining minigame bot — scaffold.

Mechanic: a mining-cart auto-scrolls right. Click once to jump, click again
mid-air to slam down. Land slams on ore deposits, avoid pits/traps.

Detector and click-policy are not implemented yet — first capture real
frames with `mining-capture`, then design ore/pit/cart detection from
those.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import check_failsafe
from common.regions import get_region
from common.session_log import session_log
from common.window import get_bounds, WindowNotFoundError
from minigames.mining.detector import find_cart, find_next_terrain

_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "assets" / "logs"

WINDOW_TITLE = "Legends Of Idleon"
POLL_INTERVAL = 0.02


def run():
    with session_log(LOGS_DIR) as log_path:
        print(f"Session log: {log_path}")
        _run_inner()


def _run_inner():
    print(f"Mining bot starting — tracking window {WINDOW_TITLE!r}. Move mouse to a corner to abort.")
    print("NOTE: detector is a stub — bot will idle until find_cart / find_next_terrain are implemented.")
    time.sleep(2)

    while True:
        check_failsafe()
        try:
            win_left, win_top, win_w, win_h = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError as e:
            print(e)
            time.sleep(1)
            continue

        play_region = get_region(_HERE, "play", win_w, win_h)
        if play_region is None:
            print("No 'play' region in regions.json. Run mining-pick-play-region first.")
            time.sleep(2)
            continue

        frame = grab_region(
            win_left + play_region["left"],
            win_top + play_region["top"],
            play_region["width"],
            play_region["height"],
        )
        cart = find_cart(frame)
        terrain = find_next_terrain(frame, cart)
        if cart is None or terrain is None:
            time.sleep(POLL_INTERVAL)
            continue

        # TODO: jump-vs-slam policy goes here once detectors are real.
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
