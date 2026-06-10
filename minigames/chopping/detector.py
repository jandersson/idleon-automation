from pathlib import Path

import cv2
import numpy as np

from common.templates import match_multiscale_center

ASSETS = Path(__file__).parent / "assets"

# OpenCV HSV: H in [0,179], S/V in [0,255]. Red wraps, so two ranges.
#
# The "pointer" is a leaf sprite that scrolls back and forth across the bar.
# Per community wisdom, the hitbox is the LEFT edge of the leaf — that's what
# we use for the zone lookup (not the leaf's center or rightmost column).
#
# Leaf is bright/saturated green (the leaf sprite). Detection happens in the
# LEAF region (above the bar), so overlap with zone-green below the bar is
# fine — they're never in the same crop. Range tuned wide to catch the leaf's
# whole body, not just the darker stem.
LEAF_HSV = ((30, 60, 60), (80, 255, 255))
GREEN_HSV = ((40, 80, 80), (80, 255, 255))         # bright zone-green
GOLD_HSV = ((20, 120, 120), (35, 255, 255))
RED_HSV_LOW = ((0, 120, 80), (10, 255, 255))
RED_HSV_HIGH = ((170, 120, 80), (179, 255, 255))


def _mask(hsv: np.ndarray, low, high) -> np.ndarray:
    return cv2.inRange(hsv, np.array(low), np.array(high))


def _column_has_color(mask: np.ndarray, x: int, min_pixels: int = 2) -> bool:
    if x < 0 or x >= mask.shape[1]:
        return False
    return int(mask[:, x].sum() // 255) >= min_pixels


def _leftmost_column(mask: np.ndarray, min_pixels_per_col: int = 2) -> int | None:
    """Return the leftmost X where the mask has at least N pixels in its column.

    Used for the leaf's left edge — that's the click hitbox per community
    wisdom. Filtering by min_pixels avoids picking up isolated noise pixels.
    """
    cols = (mask > 0).sum(axis=0)
    qualifying = np.where(cols >= min_pixels_per_col)[0]
    if len(qualifying) == 0:
        return None
    return int(qualifying[0])


def _red_columns(bar_frame: np.ndarray) -> np.ndarray:
    bar_bgr = cv2.cvtColor(bar_frame, cv2.COLOR_BGRA2BGR)
    bar_hsv = cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2HSV)
    red = cv2.bitwise_or(_mask(bar_hsv, *RED_HSV_LOW), _mask(bar_hsv, *RED_HSV_HIGH))
    return np.where((red > 0).any(axis=0))[0]


def nearest_red_distance(bar_frame: np.ndarray, x: int) -> int | None:
    """Return horizontal distance in pixels from `x` to the nearest red column.

    None if there's no red in the bar at all. Used by the bot to add a safety
    margin: clicking with the leaf too close to a red zone risks the leaf
    drifting into red during click latency.
    """
    red_cols = _red_columns(bar_frame)
    if len(red_cols) == 0:
        return None
    return int(np.min(np.abs(red_cols - x)))


def red_distance_ahead(bar_frame: np.ndarray, x: int, direction: float) -> int | None:
    """Distance in px from `x` to the nearest red column in the leaf's
    direction of travel (direction > 0 = rightward). None when no red
    lies ahead.

    The undirected nearest_red_distance is the wrong gate shape: in the
    2026-06-11 00:13 session, a chop 8px from a red BEHIND a rightward
    leaf survived while a chop 19px from a red AHEAD died — at the
    measured 257-386 px/s leaf speed, 19px ahead is only ~50-75ms of
    click latency. Risk is distance-ahead over speed, i.e. time.
    """
    red_cols = _red_columns(bar_frame)
    ahead = red_cols[red_cols >= x] if direction > 0 else red_cols[red_cols <= x]
    if len(ahead) == 0:
        return None
    return int(np.min(np.abs(ahead - x)))


def leaf_vx_px_s(
    track: list[tuple[float, int]],
    min_span_s: float = 0.02,
    max_span_s: float = 0.2,
) -> float | None:
    """Leaf horizontal velocity in px/SECOND (sign = direction) from
    recent (wall_time, x) samples; None until the usable track spans
    min_span_s. px/s, not px/poll — cadence-invariant, same units rule
    as the darts arm signal (see CLAUDE.md).

    Only samples since the leaf's most recent direction reversal are
    used. The leaf accelerates as the round progresses (game fact,
    2026-06-11), so bounces come quicker late-round and an estimate
    spanning one would smear the magnitude and could flip the sign —
    exactly when the time-to-red gate matters most. Right after a
    reversal the usable track is briefly too short and this returns
    None; the caller's pixel-margin floor covers that gap. Samples
    older than max_span_s are ignored regardless (a post-click
    cooldown gap shouldn't bridge into stale motion).
    """
    if len(track) < 2:
        return None
    # Walk backwards, keeping the run of samples moving in the newest
    # pair's direction. dx == 0 pairs carry no direction information
    # (jitter / sub-pixel motion) and don't break the run.
    direction = 0
    start = len(track) - 1
    for i in range(len(track) - 2, -1, -1):
        dx = track[i + 1][1] - track[i][1]
        if dx != 0:
            if direction == 0:
                direction = 1 if dx > 0 else -1
            elif (dx > 0) != (direction > 0):
                break
        start = i
    t1, x1 = track[-1]
    in_window = [
        (t, x) for t, x in track[start:-1] if min_span_s <= t1 - t <= max_span_s
    ]
    if not in_window:
        return None
    t0, x0 = in_window[0]  # oldest usable sample still inside the window
    return (x1 - x0) / (t1 - t0)


def analyze_bar(bar_frame: np.ndarray, leaf_frame: np.ndarray | None = None) -> tuple[int | None, str]:
    """Return (leaf_left_edge_x, zone_under_left_edge).

    leaf_frame is the strip ABOVE the bar where the leaf scrolls; it MUST be
    horizontally aligned with bar_frame (same left/right window-relative). The
    leaf's X is detected in leaf_frame and looked up against zones in bar_frame.

    If leaf_frame is None, falls back to looking for the leaf in bar_frame
    itself — for setups where leaf and bar overlap.

    zone is one of: 'green', 'gold', 'red', 'none'.
    """
    bar_bgr = cv2.cvtColor(bar_frame, cv2.COLOR_BGRA2BGR)
    bar_hsv = cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2HSV)

    leaf_source_hsv = bar_hsv
    if leaf_frame is not None:
        leaf_bgr = cv2.cvtColor(leaf_frame, cv2.COLOR_BGRA2BGR)
        leaf_source_hsv = cv2.cvtColor(leaf_bgr, cv2.COLOR_BGR2HSV)

    leaf_mask = _mask(leaf_source_hsv, *LEAF_HSV)
    leaf_x = _leftmost_column(leaf_mask)
    if leaf_x is None:
        return None, "none"

    green = _mask(bar_hsv, *GREEN_HSV)
    gold = _mask(bar_hsv, *GOLD_HSV)
    red = cv2.bitwise_or(_mask(bar_hsv, *RED_HSV_LOW), _mask(bar_hsv, *RED_HSV_HIGH))

    # Check zones in priority order: gold > green > red.
    if _column_has_color(gold, leaf_x):
        return leaf_x, "gold"
    if _column_has_color(green, leaf_x):
        return leaf_x, "green"
    if _column_has_color(red, leaf_x):
        return leaf_x, "red"
    return leaf_x, "none"


def zone_layout(bar_frame: np.ndarray, min_pixels_per_col: int = 2) -> str:
    """Run-length encoding of the bar's per-column zone colors, left to
    right — e.g. ``"n4 g58 r12 g40 o9 g70 n29"`` (g=green, o=gold,
    r=red, n=none/uncolored; the number is the run width in px).

    Logged per poll so layout dynamics across a round — zones shifting
    or shrinking between chops, gold appearing, the bar dying at round
    end — can be studied offline without frame recordings. Per-column
    priority matches analyze_bar: gold > green > red.
    """
    bgr = cv2.cvtColor(bar_frame, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    def _cols(mask: np.ndarray) -> np.ndarray:
        return (mask > 0).sum(axis=0) >= min_pixels_per_col

    red = _cols(cv2.bitwise_or(_mask(hsv, *RED_HSV_LOW), _mask(hsv, *RED_HSV_HIGH)))
    green = _cols(_mask(hsv, *GREEN_HSV))
    gold = _cols(_mask(hsv, *GOLD_HSV))

    codes = np.full(red.shape, "n", dtype="<U1")
    codes[red] = "r"        # lowest priority first — later writes win
    codes[green] = "g"
    codes[gold] = "o"

    runs: list[str] = []
    run_char, run_len = codes[0], 0
    for c in codes:
        if c == run_char:
            run_len += 1
        else:
            runs.append(f"{run_char}{run_len}")
            run_char, run_len = c, 1
    runs.append(f"{run_char}{run_len}")
    return " ".join(runs)


def find_game_over(
    frame: np.ndarray, threshold: float = 0.7
) -> tuple[bool, float]:
    """Detect the end-of-trial game-over screen via multi-scale template
    match. Returns (False, 0.0) if the template asset doesn't exist —
    capture it once via `chopping-pick-game-over`.
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


def bar_pixel_count(bar_frame: np.ndarray) -> int:
    """Count of green+red+gold pixels in the bar region. While the round
    is in progress the bar is fully colored (hundreds-to-thousands of
    pixels depending on resolution). When the round ends the bar is
    replaced with neutral UI and the count collapses to ~0.

    More reliable than template-matching a specific game-over banner —
    the bar's presence IS the game state."""
    bgr = cv2.cvtColor(bar_frame, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = _mask(hsv, *GREEN_HSV)
    gold = _mask(hsv, *GOLD_HSV)
    red = cv2.bitwise_or(_mask(hsv, *RED_HSV_LOW), _mask(hsv, *RED_HSV_HIGH))
    combined = cv2.bitwise_or(cv2.bitwise_or(green, gold), red)
    return int((combined > 0).sum())
