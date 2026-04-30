import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import click, random_delay
from common.regions import get_region
from common.session_log import session_log
from common.window import get_bounds, WindowNotFoundError
from minigames.catching.detector import find_fly, find_next_gap

_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "assets" / "logs"

WINDOW_TITLE = "Legends Of Idleon"
POLL_INTERVAL = 0.02

# Click vertical strategy: if the fly's projected Y after the next click delay
# exceeds GAP_LOWER_MARGIN px past the gap's bottom, fire a click to gain
# altitude. Tune empirically once we see real flight physics.
GAP_LOWER_MARGIN = 8

# Cooldown between consecutive clicks (avoids click-spamming if our model
# overestimates how fast the fly is dropping).
MIN_CLICK_INTERVAL = 0.05

# Where to click in the play area (Flappy Bird usually accepts clicks
# anywhere in the play region). Center of the 'play' region by default.


def run():
    with session_log(LOGS_DIR) as log_path:
        print(f"Session log: {log_path}")
        _run_inner()


def _run_inner():
    print(f"Catching bot starting — tracking window {WINDOW_TITLE!r}. Move mouse to a corner to abort.")
    time.sleep(2)

    last_click_time = 0.0
    while True:
        try:
            win_left, win_top, win_w, win_h = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError as e:
            print(e)
            time.sleep(1)
            continue

        play_region = get_region(_HERE, "play", win_w, win_h)
        if play_region is None:
            print("No 'play' region in regions.json. Run catching-pick-play-region first.")
            time.sleep(2)
            continue

        frame = grab_region(
            win_left + play_region["left"],
            win_top + play_region["top"],
            play_region["width"],
            play_region["height"],
        )
        fly_pos = find_fly(frame)
        gap = find_next_gap(frame, fly_pos)
        if fly_pos is None or gap is None:
            time.sleep(POLL_INTERVAL)
            continue

        fly_x, fly_y = fly_pos
        gap_top, gap_bottom = gap

        # Click if the fly is dropping toward / past the gap's bottom edge.
        if fly_y > gap_bottom - GAP_LOWER_MARGIN:
            now = time.time()
            if now - last_click_time >= MIN_CLICK_INTERVAL:
                print(f"fly y={fly_y}, gap=[{gap_top}..{gap_bottom}] — clicking")
                random_delay(5, 20)
                cx = play_region["left"] + play_region["width"] // 2
                cy = play_region["top"] + play_region["height"] // 2
                click(win_left + cx, win_top + cy)
                last_click_time = time.time()

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
