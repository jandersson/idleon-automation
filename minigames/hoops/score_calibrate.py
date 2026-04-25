import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.window import get_bounds
from minigames.hoops.detector import score_region
from minigames.hoops.main import SCORE_REGION_REL, WINDOW_TITLE

OUT = Path(__file__).parent / "assets" / "score_calibration.png"


def run():
    print(f"Score-region calibration. Bring the hoops minigame into view.")
    time.sleep(2)

    if SCORE_REGION_REL is None:
        print("SCORE_REGION_REL is None — score detection disabled in main.py.")
        return

    left, top, width, height = get_bounds(WINDOW_TITLE)
    print(f"Window at ({left}, {top}) size {width}x{height}")
    print(f"Score region (window-relative): {SCORE_REGION_REL}")

    frame = grab_region(left, top, width, height)
    crop = score_region(
        frame,
        SCORE_REGION_REL["left"],
        SCORE_REGION_REL["top"],
        SCORE_REGION_REL["width"],
        SCORE_REGION_REL["height"],
    )

    cv2.imwrite(str(OUT), crop)
    print(f"Wrote score-region crop to {OUT}")
    print()
    print("Open it and verify it shows the score number cleanly.")
    print(f"If the crop shows the wrong area, adjust SCORE_REGION_REL in main.py.")
