import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.templates import match_multiscale_center

ASSETS = Path(__file__).parent / "assets"


def _load(name: str) -> np.ndarray:
    path = ASSETS / name
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Template not found: {path}")
    return img


def find_release_pose(
    frame: np.ndarray, threshold: float = 0.6
) -> tuple[tuple[int, int] | None, float]:
    """Template-match the player's hand+dart in the release angle, scale-invariant.

    The hand sweeps periodically through a `)` arc. The template is captured
    at the desired release angle, so matchTemplate confidence peaks once per
    arc cycle when the hand is at that angle. Multi-scale matching means the
    same template works whether the user resizes the game window.

    Threshold lowered from 0.7 to 0.6 since multi-scale tries off-tuned scales
    and naturally peaks lower; 0.6 still discriminates the release angle from
    other arm positions.
    """
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    template = _load("release.png")
    center, val, _scale = match_multiscale_center(bgr, template)
    if val < threshold:
        return None, val
    return center, val


# score_region / score_changed live in common.score_diff. Re-exported here so
# `from minigames.darts.detector import score_region, score_changed` keeps
# working unchanged. Darts previously used a non-binarized diff with threshold
# 5.0; the common version is binarized with threshold 3.0 (same as hoops post-
# noise-fix). Keeping the same behavior is preferable since both bots crop the
# same kind of in-game UI text.
from common.score_diff import score_region, score_changed  # noqa: F401, E402
