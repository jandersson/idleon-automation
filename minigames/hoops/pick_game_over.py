"""Crop the 'Game over!' element from the live game window and save as template.

Run this when the game-over screen is currently visible. Click two corners
around the 'Game over!' text (or whatever stable element best identifies the
end-of-trial state). Saves to assets/game_over.png so find_game_over can use it.
"""
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.region_picker import pick_region
from common.window import get_bounds
from minigames.hoops.main import WINDOW_TITLE

OUT = Path(__file__).parent / "assets" / "game_over.png"


def run():
    print("Bring the game-over screen into view. Capture starts in 3s.")
    time.sleep(3)

    left, top, width, height = get_bounds(WINDOW_TITLE)
    print(f"Window at ({left}, {top}) {width}x{height}")
    region = pick_region(window_title=WINDOW_TITLE, region_name="game over")
    if region is None:
        return

    # Re-grab now so the crop matches what the user just clicked on.
    frame = grab_region(left, top, width, height)
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    crop = bgr[region["top"]:region["top"] + region["height"],
               region["left"]:region["left"] + region["width"]]
    cv2.imwrite(str(OUT), crop)
    print(f"Wrote game-over template ({crop.shape[1]}x{crop.shape[0]}) to {OUT}")


if __name__ == "__main__":
    run()
