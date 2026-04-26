"""Auto-extract a fly template from a burst of catching captures.

The fly moves between consecutive frames; the background and the rings (which
also scroll) are bigger objects. Diff consecutive frames, find the small
moving blob, crop it.

Usage:
    uv run catching-extract-fly                    # uses captures/, latest burst
    uv run catching-extract-fly --motion-threshold 30
"""
import sys
from pathlib import Path

import cv2
import numpy as np

CAPTURES = Path(__file__).parent / "assets" / "captures"
OUT = Path(__file__).parent / "assets" / "fly.png"
PREVIEW = Path(__file__).parent / "assets" / "fly_preview.png"


def _arg(name: str, default):
    if name in sys.argv:
        i = sys.argv.index(name) + 1
        if i < len(sys.argv):
            return type(default)(sys.argv[i])
    return default


def run():
    frames = sorted(CAPTURES.glob("capture_*.png"))
    if len(frames) < 3:
        print(f"Need at least 3 captures in {CAPTURES}. Run catching-capture first.")
        return

    motion_threshold = _arg("--motion-threshold", 25)
    pad = _arg("--pad", 4)
    target_idx = _arg("--frame", len(frames) // 2)
    if target_idx >= len(frames):
        target_idx = len(frames) // 2

    target_path = frames[target_idx]
    target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    if target is None:
        print(f"Could not read {target_path}")
        return

    # Per-pixel max abs diff across consecutive frame pairs.
    motion = np.zeros(target.shape[:2], dtype=np.float32)
    prev = None
    for path in frames:
        cur = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if cur is None or cur.shape != target.shape:
            continue
        if prev is not None:
            diff = cv2.absdiff(cur, prev).astype(np.float32).mean(axis=2)
            motion = np.maximum(motion, diff)
        prev = cur

    motion_mask = (motion > motion_threshold).astype(np.uint8) * 255

    # Find connected components of moving pixels. Rings scroll past too — they'd
    # show motion across the full frame width over the burst. The fly is a
    # smaller blob with motion concentrated in a tighter region. Pick the
    # SMALLEST component over a min-area threshold to avoid the rings.
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if 50 <= area <= 5000:  # exclude noise (too small) and rings (too big)
            candidates.append((area, x, y, w, h))
    if not candidates:
        print(f"No fly-shaped motion blobs found (motion_threshold={motion_threshold}).")
        print("Try lowering --motion-threshold.")
        return

    # Smallest qualifying blob is most likely the fly (rings span more area).
    candidates.sort()
    _, x, y, w, h = candidates[0]

    h_img, w_img = target.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w_img, x + w + pad)
    y1 = min(h_img, y + h + pad)
    crop = target[y0:y1, x0:x1]
    cv2.imwrite(str(OUT), crop)
    print(f"Wrote fly template ({crop.shape[1]}x{crop.shape[0]}) to {OUT}")

    preview = target.copy()
    cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.imwrite(str(PREVIEW), preview)
    print(f"Wrote preview (green box on the auto-picked fly) to {PREVIEW}")


if __name__ == "__main__":
    run()
