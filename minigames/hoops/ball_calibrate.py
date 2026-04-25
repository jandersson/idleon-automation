import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.input import click
from common.window import get_bounds
from minigames.hoops.detector import (
    BALL_HSV_LOWER,
    BALL_HSV_UPPER,
    find_hoop,
    find_platform,
)
from minigames.hoops.main import WINDOW_TITLE

OUT_DIR = Path(__file__).parent / "assets" / "ball_calibration"

BURST_FRAMES = 25
INTER_FRAME_DELAY = 0.04  # ~25 fps for ~1s of flight


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ball-calibration: bring the minigame into view. Will fire one shot in 3s and burst-capture flight frames.")
    time.sleep(3)

    left, top, width, height = get_bounds(WINDOW_TITLE)
    frame = grab_region(left, top, width, height)

    hoop_pos, hoop_conf = find_hoop(frame)
    platform_pos, _ = find_platform(frame)
    if hoop_pos is None or platform_pos is None:
        print(f"hoop_conf={hoop_conf:.2f}, platform={platform_pos} — couldn't locate both. Aborting.")
        return
    print(f"Hoop at {hoop_pos}, platform at {platform_pos}. Firing.")

    click(left + width // 2, top + height // 2)
    time.sleep(0.05)  # let the launch register

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Burst-capturing {BURST_FRAMES} frames at {INTER_FRAME_DELAY * 1000:.0f}ms intervals...")

    frames = []
    for i in range(BURST_FRAMES):
        f = grab_region(left, top, width, height)
        frames.append(f)
        time.sleep(INTER_FRAME_DELAY)

    for i, f in enumerate(frames):
        bgr = cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)
        overlay = bgr.copy()
        overlay[mask > 0] = (0, 255, 255)
        out = np.hstack([bgr, overlay])
        cv2.imwrite(str(OUT_DIR / f"flight_{stamp}_{i:02d}.png"), out)

    print(f"Wrote {BURST_FRAMES} side-by-side (raw | mask-highlighted) frames to {OUT_DIR}")
    print("Look for the basketball — should be highlighted yellow in the right pane.")
    print(f"If the ball isn't highlighted, adjust BALL_HSV_LOWER/UPPER in detector.py.")
    print(f"  Current: LOWER={BALL_HSV_LOWER.tolist()}, UPPER={BALL_HSV_UPPER.tolist()}")


if __name__ == "__main__":
    run()
