"""Dump per-colour HSV mask overlays for the fishing detector, so the
fish/mine ranges in detector.py can be tuned against a real frame.

Grabs the play region and, for each FISH_HSV colour (+ megalodon + mine),
writes <assets>/calibration/<name>.png — the play frame with that colour's
mask highlighted — plus frame.png (the raw grab). Adjust the ranges in
detector.py until each overlay lights up only the intended sprites, the
same workflow as chopping-calibrate.

Run while the fishing minigame is on screen:  fishing-calibrate
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import numpy as np

from common.capture import grab_region
from common.regions import get_region
from common.window import get_bounds, WindowNotFoundError
from minigames.fishing.detector import (
    FISH_HSV, MEGALODON_HSV_LOW, MEGALODON_HSV_HIGH, MINE_HSV, _to_hsv, _mask,
)
from minigames.fishing.main import WINDOW_TITLE

_HERE = Path(__file__).parent
OUT_DIR = _HERE / "assets" / "calibration"


def _overlay(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Original frame with the masked pixels tinted magenta."""
    out = bgr.copy()
    out[mask > 0] = (255, 0, 255)
    return cv2.addWeighted(bgr, 0.5, out, 0.5, 0)


def run():
    print("Fishing HSV calibration — capturing the play region in 3s. "
          "Make sure the fishing minigame is visible.")
    time.sleep(3)
    try:
        left, top, w, h = get_bounds(WINDOW_TITLE)
    except WindowNotFoundError as e:
        print(e)
        return
    play = get_region(_HERE, "play", w, h)
    if play is None:
        print("No 'play' region yet — run fishing-pick-play-region first.")
        return

    frame = grab_region(left + play["left"], top + play["top"], play["width"], play["height"])
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    hsv = _to_hsv(frame)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_DIR / "frame.png"), bgr)

    ranges = {name: (low, high) for name, (low, high) in FISH_HSV.items()}
    for name, (low, high) in ranges.items():
        m = _mask(hsv, low, high)
        cv2.imwrite(str(OUT_DIR / f"{name}.png"), _overlay(bgr, m))
        print(f"  {name}: {int((m > 0).sum())} px")
    meg = cv2.bitwise_or(_mask(hsv, *MEGALODON_HSV_LOW), _mask(hsv, *MEGALODON_HSV_HIGH))
    cv2.imwrite(str(OUT_DIR / "megalodon.png"), _overlay(bgr, meg))
    mine = _mask(hsv, *MINE_HSV)
    cv2.imwrite(str(OUT_DIR / "mine.png"), _overlay(bgr, mine))
    print(f"  megalodon: {int((meg > 0).sum())} px | mine: {int((mine > 0).sum())} px")
    print(f"Wrote overlays to {OUT_DIR}. Tune FISH_HSV / MINE_HSV in detector.py.")


if __name__ == "__main__":
    run()
