import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.capture import grab_region
from common.window import get_bounds
from minigames.hoops.detector import _load
from minigames.hoops.main import WINDOW_TITLE

OUT = Path(__file__).parent / "assets" / "debug_match.png"


def _annotate(frame_bgr, template, x0, y0, x1, y1, label, color):
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.shape[0] < template.shape[0] or crop.shape[1] < template.shape[1]:
        print(f"{label}: search region smaller than template — skipping")
        return
    result = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    th, tw = template.shape[:2]
    abs_x = x0 + max_loc[0]
    abs_y = y0 + max_loc[1]
    cv2.rectangle(frame_bgr, (abs_x, abs_y), (abs_x + tw, abs_y + th), color, 2)
    cv2.putText(
        frame_bgr,
        f"{label} {max_val:.2f}",
        (abs_x, max(abs_y - 6, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(frame_bgr, (x0, y0), (x1 - 1, y1 - 1), color, 1)
    print(f"{label}: best match at ({abs_x}, {abs_y}) confidence={max_val:.3f}")


def run():
    print(f"Diagnosing template matches against window {WINDOW_TITLE!r}. Bring game into view.")
    time.sleep(2)
    left, top, width, height = get_bounds(WINDOW_TITLE)
    print(f"Window: ({left}, {top}) {width}x{height}")
    frame = grab_region(left, top, width, height)
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    h, w = bgr.shape[:2]
    hoop = _load("hoop.png")
    platform = _load("platform.png")
    print(f"hoop.png: {hoop.shape[1]}x{hoop.shape[0]}, platform.png: {platform.shape[1]}x{platform.shape[0]}")

    _annotate(bgr, hoop, w // 2, 0, w, h, "hoop", (0, 165, 255))
    _annotate(bgr, platform, 0, 0, w // 2, h, "platform", (0, 255, 0))

    cv2.imwrite(str(OUT), bgr)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    run()
