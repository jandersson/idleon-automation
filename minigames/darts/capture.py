"""Burst-capture frames of the darts minigame to assets/captures/.

Used for offline analysis of arm-swing dynamics: how long the release
pose template matches per swing, what the conf trajectory looks like
across consecutive frames, etc.

Usage:
  uv run darts-capture                          # default: 30 frames @ 100ms (3s)
  uv run darts-capture --interval 0.02          # bot-poll-rate sampling (50fps)
  uv run darts-capture --interval 0.02 --count 150   # 3s burst at bot poll rate

Frames are buffered in memory during capture and written to disk
afterward so the requested interval isn't widened by per-frame PNG
encoding latency (especially noticeable at fast intervals on Windows).
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.window import get_bounds
from minigames.darts.main import WINDOW_TITLE

OUT_DIR = Path(__file__).parent / "assets" / "captures"

DEFAULT_BURST_FRAMES = 30
DEFAULT_INTER_FRAME_DELAY = 0.1  # ~3s burst at 10fps
STARTUP_DELAY = 3


def _arg(name: str, default):
    if name in sys.argv:
        i = sys.argv.index(name) + 1
        if i < len(sys.argv):
            return type(default)(sys.argv[i])
    return default


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    burst_frames = _arg("--count", DEFAULT_BURST_FRAMES)
    inter_frame_delay = _arg("--interval", DEFAULT_INTER_FRAME_DELAY)

    left, top, width, height = get_bounds(WINDOW_TITLE)
    print(f"Found window at ({left}, {top}) size {width}x{height}")
    print(f"Starting burst in {STARTUP_DELAY}s — bring the darts minigame into view now")
    time.sleep(STARTUP_DELAY)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    duration_s = burst_frames * inter_frame_delay
    print(f"Capturing {burst_frames} frames at {inter_frame_delay * 1000:.0f}ms intervals "
          f"(~{duration_s:.1f}s) — buffering in memory, writing after...")

    # Buffer in memory so PNG encoding doesn't bloat the inter-frame interval.
    frames: list = []
    t0 = time.time()
    for i in range(burst_frames):
        target_t = t0 + i * inter_frame_delay
        sleep_remaining = target_t - time.time()
        if sleep_remaining > 0:
            time.sleep(sleep_remaining)
        frame = grab_region(left, top, width, height)
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frames.append(bgr)
    actual_duration = time.time() - t0
    actual_fps = burst_frames / actual_duration if actual_duration else 0
    print(f"Captured {len(frames)} frames in {actual_duration:.2f}s ({actual_fps:.1f}fps actual)")

    for i, bgr in enumerate(frames):
        path = OUT_DIR / f"capture_{stamp}_{i:03d}.png"
        cv2.imwrite(str(path), bgr)
    print(f"Wrote {len(frames)} frames to {OUT_DIR}")


if __name__ == "__main__":
    run()
