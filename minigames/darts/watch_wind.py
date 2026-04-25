"""Snapshot the wind-indicator region while the user plays darts.

Polls the wind region at a low rate; saves a new sample only when the current
crop differs from every previously-saved sample by more than a similarity
threshold. Result: a directory of distinct wind states the user can rename to
labels like none.png, left1.png, right2.png, etc., for later classification.

Runs indefinitely — Ctrl-C to stop. While running, just play darts normally.

Usage:
  uv run darts-watch-wind
  uv run darts-watch-wind --threshold 8    # higher = stricter (more dedup)
"""
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.window import get_bounds, WindowNotFoundError
from minigames.darts.main import WINDOW_TITLE, WIND_REGION_REL

OUT_DIR = Path(__file__).parent / "assets" / "wind_samples"
POLL_SECONDS = 1.0  # how often to check; wind changes between throws


def _arg(name: str, default):
    if name in sys.argv:
        i = sys.argv.index(name) + 1
        if i < len(sys.argv):
            return type(default)(sys.argv[i])
    return default


def _crop_wind(frame_bgra) -> np.ndarray:
    bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
    r = WIND_REGION_REL
    h_img, w_img = bgr.shape[:2]
    x0 = max(0, r["left"])
    y0 = max(0, r["top"])
    x1 = min(w_img, r["left"] + r["width"])
    y1 = min(h_img, r["top"] + r["height"])
    return bgr[y0:y1, x0:x1]


def _diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 255.0
    return float(cv2.absdiff(a, b).astype(np.float32).mean())


def run():
    if WIND_REGION_REL is None:
        print("WIND_REGION_REL is None — pick it first via darts-pick-wind-region.")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    threshold = _arg("--threshold", 5.0)
    print(f"Watching wind region (Ctrl-C to stop). Dedup threshold={threshold}.")
    print(f"Saving distinct samples to {OUT_DIR}")
    print("Play normally — every distinct-looking wind state will be captured once.")

    seen: list[tuple[str, np.ndarray]] = []

    # Seed with any previously-saved samples so reruns build on prior state.
    for p in sorted(OUT_DIR.glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            seen.append((p.name, img))
    if seen:
        print(f"Loaded {len(seen)} existing samples.")

    while True:
        try:
            left, top, width, height = get_bounds(WINDOW_TITLE)
        except WindowNotFoundError as e:
            print(e)
            time.sleep(2)
            continue
        frame = grab_region(left, top, width, height)
        wind = _crop_wind(frame)

        is_new = True
        closest_diff = 999.0
        for _, ref in seen:
            d = _diff(wind, ref)
            if d < closest_diff:
                closest_diff = d
            if d < threshold:
                is_new = False
                break

        if is_new:
            stamp = datetime.now().strftime("%H%M%S")
            name = f"sample_{stamp}.png"
            cv2.imwrite(str(OUT_DIR / name), wind)
            seen.append((name, wind))
            print(f"  + new wind state: {name} (closest existing was {closest_diff:.1f} away). Total samples: {len(seen)}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    run()
