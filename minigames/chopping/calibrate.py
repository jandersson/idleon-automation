import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.regions import get_region
from common.window import get_bounds
from minigames.chopping.detector import (
    GOLD_HSV,
    GREEN_HSV,
    LEAF_HSV,
    RED_HSV_HIGH,
    RED_HSV_LOW,
)
from minigames.chopping.main import WINDOW_TITLE

_HERE = Path(__file__).parent
OUT_DIR = _HERE / "calibration"


def _mask(hsv: np.ndarray, low, high) -> np.ndarray:
    return cv2.inRange(hsv, np.array(low), np.array(high))


def run():
    OUT_DIR.mkdir(exist_ok=True)
    win_left, win_top, win_w, win_h = get_bounds(WINDOW_TITLE)
    bar_region = get_region(_HERE, "bar", win_w, win_h)
    leaf_region = get_region(_HERE, "leaf", win_w, win_h)
    if bar_region is None:
        print("No bar region in regions.json. Run chopping-pick-bar-region first.")
        return
    bar_abs = {
        "left": win_left + bar_region["left"],
        "top": win_top + bar_region["top"],
        "width": bar_region["width"],
        "height": bar_region["height"],
    }
    print(f"Calibrating in 3s — bar region resolved to {bar_abs}")
    if leaf_region is None:
        print("(No leaf region picked yet — leaf mask will use the bar crop. "
              "Run chopping-pick-leaf-region for separate leaf detection.)")
    time.sleep(3)

    bar_frame = grab_region(**bar_abs)
    bar_bgr = cv2.cvtColor(bar_frame, cv2.COLOR_BGRA2BGR)
    bar_hsv = cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2HSV)

    if leaf_region is not None:
        leaf_abs = {
            "left": win_left + leaf_region["left"],
            "top": win_top + leaf_region["top"],
            "width": leaf_region["width"],
            "height": leaf_region["height"],
        }
        leaf_frame = grab_region(**leaf_abs)
        leaf_bgr = cv2.cvtColor(leaf_frame, cv2.COLOR_BGRA2BGR)
        leaf_hsv = cv2.cvtColor(leaf_bgr, cv2.COLOR_BGR2HSV)
        cv2.imwrite(str(OUT_DIR / "00_leaf_raw.png"), leaf_bgr)
        leaf_mask = _mask(leaf_hsv, *LEAF_HSV)
        cv2.imwrite(str(OUT_DIR / "mask_leaf.png"), leaf_mask)
        overlay = leaf_bgr.copy()
        overlay[leaf_mask == 0] = 0
        cv2.imwrite(str(OUT_DIR / "overlay_leaf.png"), overlay)
    else:
        leaf_mask = _mask(bar_hsv, *LEAF_HSV)
        cv2.imwrite(str(OUT_DIR / "mask_leaf.png"), leaf_mask)
        overlay = bar_bgr.copy()
        overlay[leaf_mask == 0] = 0
        cv2.imwrite(str(OUT_DIR / "overlay_leaf.png"), overlay)

    masks = {
        "green": _mask(bar_hsv, *GREEN_HSV),
        "gold": _mask(bar_hsv, *GOLD_HSV),
        "red": cv2.bitwise_or(_mask(bar_hsv, *RED_HSV_LOW), _mask(bar_hsv, *RED_HSV_HIGH)),
    }
    cv2.imwrite(str(OUT_DIR / "00_bar_raw.png"), bar_bgr)
    for name, mask in masks.items():
        cv2.imwrite(str(OUT_DIR / f"mask_{name}.png"), mask)
        overlay = bar_bgr.copy()
        overlay[mask == 0] = 0
        cv2.imwrite(str(OUT_DIR / f"overlay_{name}.png"), overlay)

    print(f"Wrote files to {OUT_DIR}")
    print("If a mask is mostly black when it shouldn't be, widen that HSV range in detector.py")


if __name__ == "__main__":
    run()
