import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.window import get_bounds
from minigames.chopping.detector import (
    GOLD_HSV,
    GREEN_HSV,
    POINTER_HSV,
    RED_HSV_HIGH,
    RED_HSV_LOW,
)
from minigames.chopping.main import BAR_REGION_REL, WINDOW_TITLE

OUT_DIR = Path(__file__).parent / "calibration"


def _mask(hsv: np.ndarray, low, high) -> np.ndarray:
    return cv2.inRange(hsv, np.array(low), np.array(high))


def run():
    OUT_DIR.mkdir(exist_ok=True)
    win_left, win_top, _, _ = get_bounds(WINDOW_TITLE)
    abs_region = {
        "left": win_left + BAR_REGION_REL["left"],
        "top": win_top + BAR_REGION_REL["top"],
        "width": BAR_REGION_REL["width"],
        "height": BAR_REGION_REL["height"],
    }
    print(f"Calibrating in 3s — bar region resolved to {abs_region}")
    time.sleep(3)

    frame = grab_region(**abs_region)
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    masks = {
        "pointer": _mask(hsv, *POINTER_HSV),
        "green": _mask(hsv, *GREEN_HSV),
        "gold": _mask(hsv, *GOLD_HSV),
        "red": cv2.bitwise_or(_mask(hsv, *RED_HSV_LOW), _mask(hsv, *RED_HSV_HIGH)),
    }

    cv2.imwrite(str(OUT_DIR / "00_raw.png"), bgr)
    for name, mask in masks.items():
        cv2.imwrite(str(OUT_DIR / f"mask_{name}.png"), mask)
        overlay = bgr.copy()
        overlay[mask == 0] = 0
        cv2.imwrite(str(OUT_DIR / f"overlay_{name}.png"), overlay)

    print(f"Wrote {1 + 2 * len(masks)} files to {OUT_DIR}")
    print("If a mask is mostly black when it shouldn't be, widen that HSV range in detector.py")


if __name__ == "__main__":
    run()
