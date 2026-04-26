"""Catching minigame detectors.

Two things to find each frame:
- The fly (player) — its (x, y) inside the play region.
- The next hoop's gap — vertical (top, bottom) of the opening at the fly's
  rightward path.

Both are TODOs until we have captures + templates. Stubs return None so the
main loop can still run and complain in the log without crashing.
"""
from pathlib import Path

import cv2
import numpy as np

ASSETS = Path(__file__).parent / "assets"


def find_fly(frame: np.ndarray) -> tuple[int, int] | None:
    """Return (fly_x, fly_y) within the frame's coord space, or None.

    Implementation options once we have captures:
    - Template-match a fly sprite (assets/fly.png, picked via catching-pick-fly).
    - HSV-mask a unique color (if the fly is distinctly colored).
    Return the leftmost-then-topmost match center.
    """
    fly_path = ASSETS / "fly.png"
    if not fly_path.exists():
        return None
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    template = cv2.imread(str(fly_path), cv2.IMREAD_COLOR)
    if template is None:
        return None
    if bgr.shape[0] < template.shape[0] or bgr.shape[1] < template.shape[1]:
        return None
    result = cv2.matchTemplate(bgr, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < 0.6:
        return None
    th, tw = template.shape[:2]
    return (max_loc[0] + tw // 2, max_loc[1] + th // 2)


def find_next_gap(frame: np.ndarray, fly_pos: tuple[int, int] | None) -> tuple[int, int] | None:
    """Return (gap_top_y, gap_bottom_y) of the next hoop ahead of the fly, or None.

    Hoops are pairs of obstacles (one top, one bottom) with a vertical gap between.
    Strategy once we have captures:
    - Template-match the hoop edges (top and bottom sprites), pick the matched
      pair that's just to the right of fly_x, return their inner edges.
    - Or HSV-mask the hoop color, group connected components vertically.
    """
    return None
