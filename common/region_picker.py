"""Click two corners of a region in a frame; prints window-relative coords.

Usage:
    pick_region(window_title, region_name="score")

Opens a window showing the current screenshot of the named game window. User
clicks top-left, then bottom-right; the function prints a dict that can be
pasted into a per-minigame main.py constant.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.capture import grab_region
from common.window import get_bounds


def pick_region(
    window_title: str | None = None,
    region_name: str = "region",
    max_display_dim: int = 900,
    image_path: Path | None = None,
) -> dict | None:
    """Open a popup to click two corners of a region.

    Source is either a live screenshot of `window_title`, or a file at `image_path`.
    Returns the picked region coords in the source image's pixel space.
    """
    if image_path is not None:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Could not read image: {image_path}")
            return None
        print(f"Loaded {image_path.name}: {bgr.shape[1]}x{bgr.shape[0]}")
    else:
        if window_title is None:
            raise ValueError("must pass either window_title or image_path")
        left, top, width, height = get_bounds(window_title)
        print(f"Captured {window_title!r}: ({left}, {top}) {width}x{height}")
        frame = grab_region(left, top, width, height)
        bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Downscale for display only — return coords in original space.
    scale = min(1.0, max_display_dim / max(bgr.shape[:2]))
    disp_w = int(bgr.shape[1] * scale)
    disp_h = int(bgr.shape[0] * scale)
    disp = cv2.resize(bgr, (disp_w, disp_h)) if scale < 1.0 else bgr.copy()

    clicks: list[tuple[int, int]] = []

    def on_click(event, x, y, flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # Map display coords back to source.
        sx = int(round(x / scale))
        sy = int(round(y / scale))
        clicks.append((sx, sy))
        print(f"Click {len(clicks)}: ({sx}, {sy})")
        # Show feedback dot.
        cv2.circle(disp, (x, y), 5, (0, 255, 255), 2)
        if len(clicks) == 2:
            (x0, y0), (x1, y1) = clicks
            x0, x1 = sorted((x0, x1))
            y0, y1 = sorted((y0, y1))
            box_disp = (
                int(round(x0 * scale)),
                int(round(y0 * scale)),
                int(round(x1 * scale)),
                int(round(y1 * scale)),
            )
            cv2.rectangle(disp, (box_disp[0], box_disp[1]), (box_disp[2], box_disp[3]), (0, 255, 0), 2)
        cv2.imshow(win_name, disp)

    win_name = f"Pick {region_name} region (click top-left, then bottom-right; ESC to quit)"
    cv2.imshow(win_name, disp)
    cv2.setMouseCallback(win_name, on_click)
    print(f"Click two corners of the {region_name} region. ESC to abort.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            print("Aborted.")
            return None
        if len(clicks) >= 2:
            cv2.waitKey(800)  # let the green box render briefly
            cv2.destroyAllWindows()
            break

    (x0, y0), (x1, y1) = clicks
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    region = {"left": x0, "top": y0, "width": x1 - x0, "height": y1 - y0}
    print()
    print(f"  {region_name.upper()}_REGION_REL = {region}")
    return region
