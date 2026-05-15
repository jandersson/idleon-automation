"""Burst-capture frames of the mining minigame for template extraction."""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.window import get_bounds
from minigames.mining.main import WINDOW_TITLE

OUT_DIR = Path(__file__).parent / "assets" / "captures"

BURST_FRAMES = 60
INTER_FRAME_DELAY = 0.1
STARTUP_DELAY = 3


def run():
    parser = argparse.ArgumentParser(description="Burst-capture mining frames")
    parser.add_argument("--frames", type=int, default=BURST_FRAMES,
                        help=f"frame count (default {BURST_FRAMES})")
    parser.add_argument("--interval", type=float, default=INTER_FRAME_DELAY,
                        help=f"seconds between frames (default {INTER_FRAME_DELAY})")
    parser.add_argument("--startup", type=float, default=STARTUP_DELAY,
                        help=f"seconds before capture starts (default {STARTUP_DELAY})")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    left, top, width, height = get_bounds(WINDOW_TITLE)
    print(f"Found window at ({left}, {top}) size {width}x{height}")
    print(f"Starting burst in {args.startup}s — bring the mining minigame into view")
    time.sleep(args.startup)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Capturing {args.frames} frames at {args.interval * 1000:.0f}ms intervals...")
    for i in range(args.frames):
        frame = grab_region(left, top, width, height)
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(str(OUT_DIR / f"capture_{stamp}_{i:03d}.png"), bgr)
        time.sleep(args.interval)
    print(f"Wrote {args.frames} frames to {OUT_DIR}")


if __name__ == "__main__":
    run()
