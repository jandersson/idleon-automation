"""Burst-capture frames of the catching minigame for template extraction."""
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.window import get_bounds
from minigames.catching.main import WINDOW_TITLE

OUT_DIR = Path(__file__).parent / "assets" / "captures"

BURST_FRAMES = 30
INTER_FRAME_DELAY = 0.1
STARTUP_DELAY = 3


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    left, top, width, height = get_bounds(WINDOW_TITLE)
    print(f"Found window at ({left}, {top}) size {width}x{height}")
    print(f"Starting burst in {STARTUP_DELAY}s — bring the catching minigame into view")
    time.sleep(STARTUP_DELAY)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Capturing {BURST_FRAMES} frames at {INTER_FRAME_DELAY * 1000:.0f}ms intervals...")
    for i in range(BURST_FRAMES):
        frame = grab_region(left, top, width, height)
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(str(OUT_DIR / f"capture_{stamp}_{i:03d}.png"), bgr)
        time.sleep(INTER_FRAME_DELAY)
    print(f"Wrote {BURST_FRAMES} frames to {OUT_DIR}")


if __name__ == "__main__":
    run()
