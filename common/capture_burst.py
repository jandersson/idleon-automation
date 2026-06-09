"""Shared burst-capture routine behind every minigame's `capture.py` (#36).

Grabs N frames of the game window at a fixed interval and writes them to
the minigame's assets/captures/ for offline review (template extraction,
swing-dynamics analysis, region picking).

Frames are buffered in memory during capture and written to disk
afterward, with absolute-time pacing, so the requested interval isn't
widened by per-frame PNG encoding latency (noticeable at fast intervals
on Windows). This was darts' variant; it strictly dominates the
write-as-you-go loop the other bots used.
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from common.capture import grab_region
from common.window import get_bounds


def capture_burst(
    window_title: str,
    out_dir: Path,
    frame_count: int = 30,
    interval_s: float = 0.1,
    startup_delay_s: float = 3.0,
    label: str = "minigame",
    post_hint: str | None = None,
    parse_cli: bool = True,
) -> None:
    """Run one burst capture. With parse_cli, --count/--frames, --interval
    and --startup override the defaults (flag spellings kept from the
    pre-consolidation darts and mining scripts)."""
    if parse_cli:
        parser = argparse.ArgumentParser(description=f"Burst-capture {label} frames")
        parser.add_argument("--count", "--frames", dest="count", type=int,
                            default=frame_count,
                            help=f"frame count (default {frame_count})")
        parser.add_argument("--interval", type=float, default=interval_s,
                            help=f"seconds between frames (default {interval_s})")
        parser.add_argument("--startup", type=float, default=startup_delay_s,
                            help=f"seconds before capture starts (default {startup_delay_s})")
        args = parser.parse_args()
        frame_count, interval_s, startup_delay_s = args.count, args.interval, args.startup

    out_dir.mkdir(parents=True, exist_ok=True)
    left, top, width, height = get_bounds(window_title)
    print(f"Found window at ({left}, {top}) size {width}x{height}")
    print(f"Starting burst in {startup_delay_s}s — bring the {label} into view now")
    time.sleep(startup_delay_s)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    duration_s = frame_count * interval_s
    print(f"Capturing {frame_count} frames at {interval_s * 1000:.0f}ms intervals "
          f"(~{duration_s:.1f}s) — buffering in memory, writing after...")

    frames: list = []
    t0 = time.time()
    for i in range(frame_count):
        target_t = t0 + i * interval_s
        sleep_remaining = target_t - time.time()
        if sleep_remaining > 0:
            time.sleep(sleep_remaining)
        frame = grab_region(left, top, width, height)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR))
    actual_duration = time.time() - t0
    actual_fps = frame_count / actual_duration if actual_duration else 0
    print(f"Captured {len(frames)} frames in {actual_duration:.2f}s ({actual_fps:.1f}fps actual)")

    for i, bgr in enumerate(frames):
        cv2.imwrite(str(out_dir / f"capture_{stamp}_{i:03d}.png"), bgr)
    print(f"Wrote {len(frames)} frames to {out_dir}")
    if post_hint:
        print(post_hint)
