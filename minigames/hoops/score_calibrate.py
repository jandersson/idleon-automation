import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.regions import get_region
from common.window import get_bounds
from minigames.hoops.detector import score_region
from minigames.hoops.main import WINDOW_TITLE

_HERE = Path(__file__).parent
OUT = _HERE / "assets" / "score_calibration.png"


def run():
    print(f"Score-region calibration. Bring the hoops minigame into view.")
    time.sleep(2)

    left, top, win_w, win_h = get_bounds(WINDOW_TITLE)
    print(f"Window at ({left}, {top}) size {win_w}x{win_h}")
    region = get_region(_HERE, "score", win_w, win_h)
    if region is None:
        print("No 'score' region in regions.json — run hoops-pick-score-region first.")
        return
    print(f"Score region (resolved to pixel coords): {region}")

    frame = grab_region(left, top, win_w, win_h)
    crop = score_region(
        frame,
        region["left"],
        region["top"],
        region["width"],
        region["height"],
    )

    cv2.imwrite(str(OUT), crop)
    print(f"Wrote score-region crop to {OUT}")
    print()
    print("Open it and verify it shows the score number cleanly.")
    print("If the crop shows the wrong area, run hoops-pick-score-region again.")
