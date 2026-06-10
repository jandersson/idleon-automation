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


def compute_dart_dy(
    prev_bgr: np.ndarray | None,
    match_x: int,
    match_y: int,
    template: np.ndarray | None = None,
    window: int = 70,
    min_conf: float = 0.5,
) -> int | None:
    """Vertical displacement (px, positive = downward) of the dart between
    the previous frame and the current template match at (match_x, match_y).

    Fire-time pass discriminator groundwork (#26): the release template
    matches on two swing passes — the up-swing (dart moving up fast,
    launch angle >15°, 0/40 hits) and the forward release (dart level,
    20/20 hits). Pose position can't tell them apart (spawn height shifts
    both passes together), but the dart's motion between two consecutive
    frames can. Re-matches the template at fixed scale inside a window
    around the current match in the PREVIOUS frame — which the main loop
    already holds for arm-motion computation, so this costs one
    small-crop matchTemplate and adds no extra capture before the click.

    Returns match_y(cur) - prev_match_y, or None when there is no
    previous frame or the dart isn't confidently found in the window
    (swung in from outside it, or the live match scale is far from the
    template's native scale).
    """
    if prev_bgr is None:
        return None
    if template is None:
        template = _load("release.png")
    h, w = prev_bgr.shape[:2]
    x0, x1 = max(0, match_x - window), min(w, match_x + window)
    y0, y1 = max(0, match_y - window), min(h, match_y + window)
    crop = prev_bgr[y0:y1, x0:x1]
    if crop.shape[0] < template.shape[0] or crop.shape[1] < template.shape[1]:
        return None
    res = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < min_conf:
        return None
    prev_cy = y0 + max_loc[1] + template.shape[0] // 2
    return match_y - prev_cy


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
    h, w = bgr.shape[:2]
    template = _load("release.png")
    # Left ~65% only: the darts stuck in the board (x ~900+) match the
    # dart-shaft template at a constant ~0.60 and were winning the
    # best-match slot over genuine sub-threshold poses (observed during
    # the 2026-06-10 12:01 stall). Player spawns range x ~350-450.
    center, val, _scale = match_multiscale_center(
        bgr, template, region=(0, 0, int(w * 0.65), h)
    )
    if val < threshold:
        return None, val
    return center, val


def find_celebration(
    frame: np.ndarray, threshold: float = 0.75
) -> tuple[bool, float]:
    """Detect the streak celebration banner ("...one hundred and
    EIGHTY!", observed at 4 bullseyes in a row, 2026-06-10). While it
    plays, the player sprite celebrates instead of swinging, so the
    release pose can't match and the no-pose timeout would misread the
    state as game over. Template is the stable middle words of the
    bottom-center text (the Ooo/IIGHTY ends may stretch with streak
    size); validated 1.0 on the celebration frame vs ~0.34 floor on
    normal play. Returns (False, 0.0) when the template isn't captured.
    """
    path = ASSETS / "celebration.png"
    if not path.exists():
        return False, 0.0
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    template = _load("celebration.png")
    center, val, _scale = match_multiscale_center(bgr, template)
    if val < threshold:
        return False, val
    return True, val


def find_game_over(
    frame: np.ndarray, threshold: float = 0.85
) -> tuple[bool, float]:
    """Detect the end-of-game screen via multi-scale template match.

    Returns (False, 0.0) if the template hasn't been captured yet — the
    main loop falls back to its no-pose timeout heuristic in that case.
    Capture the template via `darts-pick-game-over` while the game-over
    screen is visible.

    Threshold bumped 2026-05-24 from 0.7 to 0.85 after a session
    ended at conf=0.76 with the player visibly still in the dartboard
    scene — startup phase has no life cap (player keeps throwing until
    a hit), so a 7-throw all-miss session can't legitimately end in
    game-over. The earlier 0.7 was leaving ~6px of headroom against
    false positives during normal play.
    """
    path = ASSETS / "game_over.png"
    if not path.exists():
        return False, 0.0
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    template = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if template is None:
        return False, 0.0
    _, val, _scale = match_multiscale_center(bgr, template)
    return val >= threshold, val


# score_region / score_changed live in common.score_diff. Re-exported here so
# `from minigames.darts.detector import score_region, score_changed` keeps
# working unchanged. Darts previously used a non-binarized diff with threshold
# 5.0; the common version is binarized with threshold 3.0 (same as hoops post-
# noise-fix). Keeping the same behavior is preferable since both bots crop the
# same kind of in-game UI text.
from common.score_diff import score_region, score_changed  # noqa: E402

# Explicit re-export so pyflakes doesn't flag the imports as unused.
__all__ = ["score_region", "score_changed"]
