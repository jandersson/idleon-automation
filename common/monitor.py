"""Per-shot monitor folder helpers.

Game-mode minigames (hoops, darts) save a folder per shot under
assets/monitor/ with pre/post screenshots, optional flight frames, and a
metadata file. Lets the user (or future Claude) review what the bot saw on
disk after a session, since the bot can't watch the screen live.

Each minigame creates its own MONITOR_DIR; this module just provides the
shared file layout.
"""
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def make_shot_dir(monitor_dir: Path, shot_idx: int, prefix: str = "shot") -> Path:
    """Create assets/monitor/<prefix>_<idx>_<HHMMSS>/ and return its path."""
    monitor_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%H%M%S")
    sub = monitor_dir / f"{prefix}_{shot_idx:03d}_{stamp}"
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def save_frame(path: Path, frame_bgra: np.ndarray) -> None:
    """Convert BGRA → BGR and write to path. Caller picks the filename."""
    bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
    cv2.imwrite(str(path), bgr)


def save_meta(path: Path, **fields) -> None:
    """Write a meta.txt-style file with one `key=value` per line."""
    lines = [f"{k}={v}" for k, v in fields.items()]
    path.write_text("\n".join(lines), encoding="utf-8")
