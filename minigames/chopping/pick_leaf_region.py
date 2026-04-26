"""Click two corners around the LEAF'S TRACK (the strip above the bar where
the leaf scrolls back and forth). Saves to regions.json as fractions.

The leaf and bar are different regions: leaf for X-position detection,
bar for zone-color lookup at that X.
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
    print("Pick the LEAF region — the strip ABOVE the bar where the leaf scrolls.")
    print("Capture starts in 3s.")
    time.sleep(3)
    region = pick_region(window_title=WINDOW_TITLE, region_name="leaf")
    if region is None:
        return
    _, _, win_w, win_h = get_bounds(WINDOW_TITLE)
    path = save_region(MINIGAME_DIR, "leaf", region, win_w, win_h)
    print(f"Saved leaf region (as fractions of {win_w}x{win_h}) to {path}: {region}")


if __name__ == "__main__":
    run()
