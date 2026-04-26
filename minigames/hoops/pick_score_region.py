"""Click two corners around the score number; saves to regions.json as
fractions of the current window size so it survives resizing.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.region_picker import pick_region
from common.regions import save_region
from common.window import get_bounds
from minigames.hoops.main import WINDOW_TITLE

MINIGAME_DIR = Path(__file__).parent


def run():
    print("Pick the score region. Capture starts in 3s.")
    time.sleep(3)
    region = pick_region(window_title=WINDOW_TITLE, region_name="score")
    if region is None:
        return
    _, _, win_w, win_h = get_bounds(WINDOW_TITLE)
    path = save_region(MINIGAME_DIR, "score", region, win_w, win_h)
    print(f"Saved score region (as fractions of {win_w}x{win_h}) to {path}: {region}")


if __name__ == "__main__":
    run()
