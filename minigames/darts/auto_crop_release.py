"""Auto-crop the player's arm+dart from a chosen capture frame.

Uses the burst-capture sequence to identify which pixels MOVE across frames
(= the swept arm + dart), then intersects with the player's white-body mask
in the chosen frame to get a tight crop of the arm in its current pose.

Usage:
  uv run darts-auto-crop-release                 # uses frame index 8
  uv run darts-auto-crop-release --frame 12      # try a different frame index
  uv run darts-auto-crop-release --frame 12 --motion-threshold 30
"""
import sys
from pathlib import Path

import cv2
import numpy as np

CAPTURES = Path(__file__).parent / "assets" / "captures"
OUT = Path(__file__).parent / "assets" / "release.png"
PREVIEW = Path(__file__).parent / "assets" / "release_preview.png"


def _arg(name: str, default):
    if name in sys.argv:
        i = sys.argv.index(name) + 1
        if i < len(sys.argv):
            return type(default)(sys.argv[i])
    return default


def run():
    frames = sorted(CAPTURES.glob("capture_*.png"))
    if len(frames) < 2:
        print(f"Need at least 2 captures in {CAPTURES}. Run darts-capture first.")
        return

    target_idx = _arg("--frame", 8)
    motion_threshold = _arg("--motion-threshold", 20)
    pad = _arg("--pad", 4)
    if target_idx >= len(frames):
        print(f"Frame index {target_idx} out of range (0..{len(frames) - 1})")
        return
    target_path = frames[target_idx]
    print(f"Target frame: {target_path.name}")

    target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    if target is None:
        print(f"Could not read {target_path}")
        return

    # Motion mask: per-pixel max abs-diff across consecutive frame pairs.
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

    # White-ish player pixels in the target frame.
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 70, 255]))

    arm_mask = cv2.bitwise_and(motion_mask, white_mask)
    coords = cv2.findNonZero(arm_mask)
    if coords is None:
        print(f"No arm pixels found (motion_threshold={motion_threshold}).")
        print("Try lowering --motion-threshold, or use a different --frame.")
        return

    x, y, w, h = cv2.boundingRect(coords)
    h_img, w_img = target.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w_img, x + w + pad)
    y1 = min(h_img, y + h + pad)

    crop = target[y0:y1, x0:x1]
    cv2.imwrite(str(OUT), crop)
    print(f"Wrote release template ({crop.shape[1]}x{crop.shape[0]}) to {OUT}")

    # Preview: target frame with a green box drawn on the picked region.
    preview = target.copy()
    cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.imwrite(str(PREVIEW), preview)
    print(f"Wrote preview (green box on full frame) to {PREVIEW}")


if __name__ == "__main__":
    run()
