import math
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


def _leaf_left_edge(mask: np.ndarray, min_pixels_per_col: int = 2) -> int | None:
    """Left edge of the WIDEST qualifying column run — the leaf body.

    The absolute leftmost qualifying column (the old rule) breaks when
    anything else leaf-green intrudes on the strip: in the 2026-07-11
    environment the bar's green end cap poked a few 1-2px columns into
    the strip's left edge and pinned the 'leaf' at x=4 while the real
    ~10px-wide sprite sat at x=75. The widest run is the sprite; its
    left edge is the hitbox as before. Gaps of 1px bridge sprite
    outlines."""
    cols = (mask > 0).sum(axis=0)
    qualifying = np.where(cols >= min_pixels_per_col)[0]
    if len(qualifying) == 0:
        return None
    splits = np.where(np.diff(qualifying) > 2)[0]
    starts = [0, *(splits + 1)]
    ends = [*splits, len(qualifying) - 1]
    widths = [qualifying[e] - qualifying[s] for s, e in zip(starts, ends)]
    k = int(np.argmax(widths))
    return int(qualifying[starts[k]])


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


def _distance_ahead(cols: np.ndarray, x: int, direction: float) -> int | None:
    ahead = cols[cols >= x] if direction > 0 else cols[cols <= x]
    if len(ahead) == 0:
        return None
    return int(np.min(np.abs(ahead - x)))


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
    return _distance_ahead(_red_columns(bar_frame), x, direction)


def gold_distance_ahead(bar_frame: np.ndarray, x: int, direction: float) -> int | None:
    """Distance in px from `x` to the nearest gold column in the leaf's
    direction of travel; None when no gold lies ahead. Powers the
    same-sweep gold upgrade: gold pays +2 AND slows the leaf
    (docs/chopping_notes.md), so a safe green fire is deferred when
    gold lies ahead of the leaf before any red.
    """
    bar_bgr = cv2.cvtColor(bar_frame, cv2.COLOR_BGRA2BGR)
    bar_hsv = cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2HSV)
    gold_cols = np.where((_mask(bar_hsv, *GOLD_HSV) > 0).any(axis=0))[0]
    return _distance_ahead(gold_cols, x, direction)


# Minimum sin(theta) for a sample to count in infer_vmax. Near the bar
# edges sin(theta) -> 0: the per-sample V_max estimate explodes on
# jitter, and the leaf also spends MOST of its time there (eased motion
# lingers at the turnarounds) — including edge samples with a floored
# correction biased the median ~25% low and let chop 14 (10:51 session)
# fire at a priced 138ms when honest sweep peaks of 600-840 px/s made
# it ~103ms. Mid-region samples are direct V_max evidence; use only
# those.
MIN_SIN_FOR_VMAX = 0.5


VMAX_PERCENTILE = 0.75


def infer_vmax(
    samples: list[tuple[int, float]], bar_w: int,
    percentile: float = VMAX_PERCENTILE,
) -> float | None:
    """Robust sweep peak speed from recent (x, |vx|) samples.

    Under the eased sweep model (x(θ) = (W/2)(1−cosθ), v = V_max·sinθ),
    each sample in the mid region (sin(θ) ≥ MIN_SIN_FOR_VMAX) implies
    V_max = |vx| / sin(θ(x)); edge samples are EXCLUDED (see the
    constant's comment — flooring them dragged the estimate low).

    The estimate is the 75th percentile of the corrected samples, not
    the median: the kill-relevant speed is what the leaf CAN do, and
    with the per-chop ramp the trailing window mixes slower old sweeps
    with faster new ones — the median lags that ramp (chop 14, 10:51
    session: median priced the fatal window at 141ms; honest sweep
    peaks of 600-840 px/s made it ~104ms). p75 over ≥5 samples still
    can't be moved by the 1600+ px/s jitter spikes that poisoned the
    raw-max estimator (the 01:39 starve) — spikes are a few samples
    out of dozens. None with fewer than 5 usable samples.
    """
    if bar_w <= 0:
        return None
    ests = []
    for x, v in samples:
        c = 1.0 - 2.0 * x / bar_w
        s = math.sqrt(max(0.0, 1.0 - c * c))
        if s >= MIN_SIN_FOR_VMAX:
            ests.append(v / s)
    if len(ests) < 5:
        return None
    ests.sort()
    return ests[min(len(ests) - 1, int(len(ests) * percentile))]


def eased_time_to_x_s(
    x: float, direction: float, distance: float, v_max: float, bar_w: int
) -> float | None:
    """Seconds for the leaf to cover `distance` px in `direction`, under
    the eased sweep model — position-aware, unlike
    distance/instantaneous-speed: a leaf in the slow edge region takes
    longer to reach a far point than the linear estimate says (it has
    to accelerate first), and a mid-bar leaf reaches nearby points
    sooner (it's already near peak speed). x(θ) = (W/2)(1−cosθ) gives
    θ(p) = arccos(1 − 2p/W) and time = Δθ / ω with ω = 2·V_max/W.

    Mirrors x for leftward motion so the math is direction-free.
    Returns None when v_max/bar_w are unusable. Distances beyond the
    bar end clamp to the turnaround (the sweep's quarter period).
    """
    if v_max <= 0 or bar_w <= 0:
        return None

    def theta(p: float) -> float:
        return math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * p / bar_w)))

    xn = float(x) if direction > 0 else float(bar_w - x)
    target = min(float(bar_w), xn + distance)
    omega = 2.0 * v_max / bar_w
    return max(0.0, theta(target) - theta(xn)) / omega


def eased_time_to_red_ms(
    x: int, direction: float, red_ahead: int, v_max: float, bar_w: int
) -> int | None:
    """Time (ms) to the nearest red in `direction` — the fire gate's
    original entry point; thin wrapper over eased_time_to_x_s."""
    t = eased_time_to_x_s(x, direction, red_ahead, v_max, bar_w)
    return None if t is None else int(t * 1000)


def gold_window_ms(bar_frame: np.ndarray, v_max: float) -> int | None:
    """Best achievable time-to-red (ms) when ENTERING the gold zone, over
    both travel directions, under the eased model at v_max — i.e. the
    most the fire gate could ever be offered on this layout's gold.

    Prices a gold chase before it starts: the layout only re-rolls on a
    chop, so a red-flanked gold that can't reach MIN_TIME_TO_RED_MS now
    never will (speed only rises) — chasing it just burns the chase
    deadline every re-arm (2026-07-06 21:01 run: four 6s+ waits, zero
    conversions, ~28s, on an 'r11 o33 r19' sandwich at ~600 px/s).

    Returns None when the bar has no gold. A direction with no red
    ahead of the gold entry is an unbounded window (returns a large
    value). Assumes one contiguous gold zone (all layouts observed so
    far); with several, this prices the outermost entries only.
    """
    bar_bgr = cv2.cvtColor(bar_frame, cv2.COLOR_BGRA2BGR)
    bar_hsv = cv2.cvtColor(bar_bgr, cv2.COLOR_BGR2HSV)
    gold_cols = np.where((_mask(bar_hsv, *GOLD_HSV) > 0).any(axis=0))[0]
    if len(gold_cols) == 0:
        return None
    red_cols = _red_columns(bar_frame)
    bar_w = bar_frame.shape[1]
    best = None
    for direction, entry in ((1.0, int(gold_cols.min())),
                             (-1.0, int(gold_cols.max()))):
        red_ahead = _distance_ahead(red_cols, entry, direction)
        if red_ahead is None:
            return 10**6  # no red past the gold in this direction
        t = eased_time_to_red_ms(entry, direction, red_ahead, v_max, bar_w)
        if t is not None and (best is None or t > best):
            best = t
    return best


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
    leaf_x = _leaf_left_edge(leaf_mask)
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


# ---- Overlay auto-location (2026-07-11) -----------------------------
# The minigame overlay is anchored above the PLAYER, so its screen
# position changes per environment (a new-map run pointed the picked
# regions at scenery and the bot exited on 'bar dead' in 4s). Same
# lesson as the Play button in CLAUDE.md: in-world UI needs visual
# detection, not coordinate caching. The bar is the most findable
# object on screen — a wide, thin horizontal band where zone green and
# red coexist — and every other region hangs off it at fixed offsets
# (measured from the hand-picked regions.json of the original
# environment, expressed in bar-width units so they scale).

# Band acceptance thresholds, as fractions of the window width where
# noted. The bar is ~23% of the window wide; green-only bars (XP), the
# red-only HP bar, portals, buttons and sprites all fail the
# both-colors + width + aspect tests.
BAR_MIN_WIDTH_FRAC = 0.12
BAR_MAX_WIDTH_FRAC = 0.40
BAR_MIN_H, BAR_MAX_H = 5, 40
BAR_MIN_FILL = 0.75          # colored cols / span inside the band
BAR_MIN_GREEN_FRAC = 0.15    # of the span
BAR_MIN_RED_FRAC = 0.03


def find_bar_rect(window_frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Locate the minigame bar in a full-window BGRA frame.

    Returns (left, top, width, height) window-relative, or None when no
    band matches. Signature: a horizontal band of rows dense in zone
    colors, wide-but-thin, whose columns are mostly contiguous and
    contain BOTH green and red (gold counts as green for fill).
    """
    bgr = cv2.cvtColor(window_frame, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    green = _mask(hsv, *GREEN_HSV)
    gold = _mask(hsv, *GOLD_HSV)
    red = cv2.bitwise_or(_mask(hsv, *RED_HSV_LOW), _mask(hsv, *RED_HSV_HIGH))
    combined = cv2.bitwise_or(cv2.bitwise_or(green, gold), red)

    win_h, win_w = combined.shape
    min_w = int(BAR_MIN_WIDTH_FRAC * win_w)
    row_counts = (combined > 0).sum(axis=1)
    dense = row_counts >= min_w

    best: tuple[int, int, int, int] | None = None
    best_span = 0
    y = 0
    while y < win_h:
        if not dense[y]:
            y += 1
            continue
        y0 = y
        while y < win_h and dense[y]:
            y += 1
        y1 = y  # [y0, y1) is a candidate band
        band_h = y1 - y0
        if not (BAR_MIN_H <= band_h <= BAR_MAX_H):
            continue
        band = combined[y0:y1]
        cols = np.where((band > 0).any(axis=0))[0]
        if len(cols) == 0:
            continue
        # Evaluate contiguous column RUNS (gaps <= 1px only: real zones
        # touch each other directly), not the raw min-max span: sprites
        # or the Chop button sharing the band's rows would otherwise
        # stretch the span and sink the fill test, and the bar's own
        # decorative green END CAPS sit 2-3px off the playfield — the
        # tight gap rule splits them away so the rect is the playfield
        # alone (caps above the strip otherwise pollute the leaf mask).
        splits = np.where(np.diff(cols) > 1)[0]
        run_bounds = zip(
            [0, *(splits + 1)], [*splits, len(cols) - 1])
        for i0, i1 in run_bounds:
            x0, x1 = int(cols[i0]), int(cols[i1]) + 1
            span = x1 - x0
            if not (min_w <= span <= BAR_MAX_WIDTH_FRAC * win_w):
                continue
            if (i1 - i0 + 1) / span < BAR_MIN_FILL:
                continue
            green_cols = int(((green[y0:y1, x0:x1] > 0).any(axis=0)
                              | (gold[y0:y1, x0:x1] > 0).any(axis=0)).sum())
            red_cols = int((red[y0:y1, x0:x1] > 0).any(axis=0).sum())
            if green_cols < BAR_MIN_GREEN_FRAC * span:
                continue
            if red_cols < BAR_MIN_RED_FRAC * span:
                continue
            if span > best_span:
                best = (x0, y0, span, band_h)
                best_span = span
    return best


# Companion-region offsets in units of bar WIDTH (the most stably
# measured bar dimension), calibrated from the original environment's
# hand-picked regions.json (window 960x572: bar (76,121,222,18), leaf
# (75,102,222,19), button (310,121,49,24), score (74,140,32,13)) and
# verified against a second environment's capture (2026-07-11: every
# derived region landed on its overlay element).
LEAF_H_FRAC = 19 / 222
BUTTON_DX_FRAC = 234 / 222   # bar left -> button left
BUTTON_W_FRAC = 49 / 222
BUTTON_H_FRAC = 24 / 222
SCORE_DX_FRAC = -2 / 222     # bar left -> score left
SCORE_W_FRAC = 32 / 222
SCORE_H_FRAC = 13 / 222


def derive_overlay_regions(
    bar_rect: tuple[int, int, int, int],
) -> dict[str, dict[str, int]]:
    """Window-relative regions for the whole overlay from the bar rect:
    the leaf strip directly above the bar, the Chop button to its
    right, and the PTS counter under its left end. Same dict shape as
    common.regions.get_region returns."""
    left, top, w, h = bar_rect
    leaf_h = max(1, round(LEAF_H_FRAC * w))
    return {
        "bar": {"left": left, "top": top, "width": w, "height": h},
        "leaf": {"left": left, "top": top - leaf_h, "width": w, "height": leaf_h},
        "button": {
            "left": left + round(BUTTON_DX_FRAC * w), "top": top,
            "width": round(BUTTON_W_FRAC * w), "height": round(BUTTON_H_FRAC * w),
        },
        "score": {
            "left": left + round(SCORE_DX_FRAC * w), "top": top + h + 1,
            "width": round(SCORE_W_FRAC * w), "height": round(SCORE_H_FRAC * w),
        },
    }


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
