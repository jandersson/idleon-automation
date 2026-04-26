"""Click two corners around the play area (the rectangle where the fly flies
and hoops scroll). Saves to regions.json as fractions.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.region_picker import pick_region
from common.regions import save_region
from common.window import get_bounds
from minigames.catching.main import WINDOW_TITLE

MINIGAME_DIR = Path(__file__).parent


def run():
    print("Pick the catching play region. Capture starts in 3s.")
    time.sleep(3)
    region = pick_region(window_title=WINDOW_TITLE, region_name="play")
    if region is None:
        return
    _, _, win_w, win_h = get_bounds(WINDOW_TITLE)
    path = save_region(MINIGAME_DIR, "play", region, win_w, win_h)
    print(f"Saved play region (as fractions of {win_w}x{win_h}) to {path}: {region}")


if __name__ == "__main__":
    run()
