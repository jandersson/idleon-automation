import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import click, random_delay
from common.regions import get_region
from common.session_log import session_log
from common.window import get_bounds, WindowNotFoundError
from minigames.chopping.detector import analyze_bar

_HERE = Path(__file__).parent
LOGS_DIR = _HERE / "assets" / "logs"

WINDOW_TITLE = "Legends Of Idleon"

# Regions are loaded from assets/regions.json each iteration so they survive
# window resizes. Pick via chopping-pick-bar-region / chopping-pick-button-region.
# - bar: the colored zone strip with the sliding pointer
# - button: the CHOP button (bot clicks its center)

POLL_INTERVAL = 0.01
COOLDOWN_AFTER_CLICK = 0.25


def run():
    with session_log(LOGS_DIR) as log_path:
        print(f"Session log: {log_path}")
        _run_inner()


def _run_inner():
    print(f"Chopping bot starting — tracking window {WINDOW_TITLE!r}. Move mouse to a corner to abort.")
    time.sleep(2)

    while True:
        try:
            win_left, win_top, win_w, win_h = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError as e:
            print(e)
            time.sleep(1)
            continue

        bar_region = get_region(_HERE, "bar", win_w, win_h)
        leaf_region = get_region(_HERE, "leaf", win_w, win_h)
        button_region = get_region(_HERE, "button", win_w, win_h)
        if bar_region is None or button_region is None:
            print("Missing region(s) in regions.json. Run chopping-pick-bar-region and chopping-pick-button-region first.")
            time.sleep(2)
            continue

        bar_frame = grab_region(
            win_left + bar_region["left"],
            win_top + bar_region["top"],
            bar_region["width"],
            bar_region["height"],
        )
        # Leaf region is optional: if not picked, fall back to detecting the
        # leaf in the bar region itself (works when they visually overlap).
        leaf_frame = None
        if leaf_region is not None:
            leaf_frame = grab_region(
                win_left + leaf_region["left"],
                win_top + leaf_region["top"],
                leaf_region["width"],
                leaf_region["height"],
            )
        pointer_x, zone = analyze_bar(bar_frame, leaf_frame=leaf_frame)

        if pointer_x is not None and zone in ("green", "gold"):
            print(f"Pointer at x={pointer_x} in {zone} zone — chopping")
            random_delay(20, 60)
            button_cx = button_region["left"] + button_region["width"] // 2
            button_cy = button_region["top"] + button_region["height"] // 2
            click(win_left + button_cx, win_top + button_cy)
            time.sleep(COOLDOWN_AFTER_CLICK)
            continue

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
