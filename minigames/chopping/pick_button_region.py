"""Click two corners around the CHOP button; the bot will click its center.
Saves to regions.json as fractions of current window size.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.region_picker import pick_region
from common.regions import save_region
from common.window import get_bounds
from minigames.chopping.main import WINDOW_TITLE

MINIGAME_DIR = Path(__file__).parent


def run():
    print("Pick the CHOP button region. Capture starts in 3s.")
    time.sleep(3)
    region = pick_region(window_title=WINDOW_TITLE, region_name="button")
    if region is None:
        return
    _, _, win_w, win_h = get_bounds(WINDOW_TITLE)
    path = save_region(MINIGAME_DIR, "button", region, win_w, win_h)
    print(f"Saved button region (as fractions of {win_w}x{win_h}) to {path}: {region}")


if __name__ == "__main__":
    run()
