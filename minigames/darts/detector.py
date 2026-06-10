import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.templates import ScaleLockMatcher, match_multiscale_center

ASSETS = Path(__file__).parent / "assets"

# Templates are immutable for the lifetime of a bot process (capture
# tooling writes them between sessions), and the poll loop calls the
# detectors every ~250ms — re-reading PNGs from disk each call was
# measurable tick cost. Missing optional templates are NOT cached so a
# capture taken mid-session is picked up on the next poll.
_TEMPLATE_CACHE: dict[str, np.ndarray] = {}


def _load(name: str) -> np.ndarray:
    cached = _TEMPLATE_CACHE.get(name)
    if cached is not None:
        return cached
    path = ASSETS / name
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Template not found: {path}")
    _TEMPLATE_CACHE[name] = img
    return img


def _load_optional(name: str) -> np.ndarray | None:
    """Like _load but returns None when the template hasn't been
    captured yet, or exists but is unreadable (game_over.png,
    celebration.png)."""
    if name in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[name]
    img = cv2.imread(str(ASSETS / name), cv2.IMREAD_COLOR)
    if img is not None:
        _TEMPLATE_CACHE[name] = img
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


# Scale-locked matcher for the per-poll release-pose match (poll-loop
# step 3, 2026-06-10): genuine poses win at scales 0.9-1.1 (428/511
# at-fire frames) and the winning scale only moves on window resize, so
# most polls skip the rest of the sweep — 78ms -> 47ms measured, with
# 2000/2029 polls returning bit-identical (conf, center) to the full
# sweep on the recorded corpus (scripts/validate_pose_scale_lock.py).
# always_scales=(0.6,) keeps the sub-threshold junk floor (which wins
# at 0.6 in ~97% of pose-absent frames, incl. the (146,38) phantom at
# 0.5633) reading the same as before, so the adaptive-threshold and
# best_recent_conf semantics in main.py are unchanged on locked polls.
_POSE_MATCHER = ScaleLockMatcher(always_scales=(0.6,))


def find_release_pose(
    frame: np.ndarray, threshold: float = 0.6
) -> tuple[tuple[int, int] | None, float]:
    """Template-match the player's hand+dart in the release angle, scale-invariant.

    The hand sweeps periodically through a `)` arc. The template is captured
    at the desired release angle, so matchTemplate confidence peaks once per
    arc cycle when the hand is at that angle. Multi-scale matching means the
    same template works whether the user resizes the game window — the
    scale lock re-syncs via a full sweep every resync_every polls.

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
    center, val, _scale = _POSE_MATCHER.match_center(
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
    template = _load_optional("celebration.png")
    if template is None:
        return False, 0.0
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    center, val, _scale = match_multiscale_center(bgr, template)
    if val < threshold:
        return False, val
    return True, val


# Coarse-pass gate for the game-over cascade. Calibrated 2026-06-10
# against 12 real frames: the half-res noise floor on normal play /
# overlay screens is 0.48-0.54 (NOTE: higher than the full-res ~0.43
# floor — downscaling smooths away the detail that kept the full-res
# correlation low), while a true banner scores 1.000 at half res. 0.7
# splits that with wide margin. If a real game-over ever slipped under
# it, the no-pose timeout heuristic still ends the session — the
# cascade affects how fast, not whether.
GAME_OVER_COARSE_GATE = 0.7
_GAME_OVER_COARSE_SCALE = 0.5


def find_game_over(
    frame: np.ndarray, threshold: float = 0.85
) -> tuple[bool, float]:
    """Detect the end-of-game screen via multi-scale template match.

    Returns (False, 0.0) if the template hasn't been captured yet — the
    main loop falls back to its no-pose timeout heuristic in that case.
    Capture the template via `darts-pick-game-over` while the game-over
    screen is visible.

    Coarse-to-fine (2026-06-10): this runs on EVERY poll and the
    full-window multi-scale match was 115ms — 57% of the compute-bound
    poll tick. The half-resolution prefilter does ~1/16 of the
    matchTemplate work (both operands' areas quarter); only when it
    clears GAME_OVER_COARSE_GATE does the full-resolution match run
    and decide. Normal play exits at the coarse pass, reporting conf
    0.0 — half-res conf values run ~0.1 higher than full-res ones, so
    leaking them to callers would silently shift the overlay probe's
    `go_conf < OVERLAY_PROBE_GO_CONF_MAX` comparison; 0.0 states what
    the coarse exit means ("clearly not game over") on the full-res
    scale the callers expect.

    Threshold bumped 2026-05-24 from 0.7 to 0.85 after a session
    ended at conf=0.76 with the player visibly still in the dartboard
    scene — startup phase has no life cap (player keeps throwing until
    a hit), so a 7-throw all-miss session can't legitimately end in
    game-over. The earlier 0.7 was leaving ~6px of headroom against
    false positives during normal play.
    """
    template = _load_optional("game_over.png")
    if template is None:
        return False, 0.0
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    s = _GAME_OVER_COARSE_SCALE
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    small_t = _TEMPLATE_CACHE.get("game_over.png@coarse")
    if small_t is None:
        small_t = cv2.resize(template, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        _TEMPLATE_CACHE["game_over.png@coarse"] = small_t
    _, coarse_val, _scale = match_multiscale_center(small, small_t)
    if coarse_val < GAME_OVER_COARSE_GATE:
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
