"""Fishing-minigame CV — pure functions, no IO.

Detects the cast targets in the play region: the coloured fish (Green /
Eel / Squid / Whale / Megalodon) and the mines, by HSV colour masking +
blob centroids — the same style as the chopping bot (gold>green>red),
because the minigame distinguishes targets purely by colour.

!! CALIBRATION REQUIRED !! The HSV ranges below are first-guess starting
points from the wiki sprite colours (MGgreenfish/MGyellowfish/
MGpurplefish/MGbluefish/MGredfish, MGfsh5 mine). They WILL need tuning
against real frames before detection is reliable — run `fishing-observe`
to capture frames, then `fishing-calibrate` to dump per-colour mask
overlays and adjust. Until then `find_fish` may under/over-detect.

Frame format: mss.grab returns BGRA; convert BGRA2BGR then BGR2HSV first
(the repo-wide gotcha).
"""
from pathlib import Path

import cv2
import numpy as np

from common.templates import match_multiscale_center, match_multiscale_zncc_center

ASSETS = Path(__file__).parent / "assets"

# OpenCV HSV: H in [0,179], S/V in [0,255]. Red wraps -> two ranges.
# Calibrated 2026-06-17 from the wiki minigame sprites (MGgreenfish/yellow/
# purple/blue/red, MGfsh5 mine) + a live observe session (960x572). `green`
# is verified in-game (H77-82); the rest are sprite-derived and still want a
# live check when they appear (the streak gates eel@3 / squid@6 / whale@13).
# Detection is restricted to the cast bar (find_cast_bar) — over the raw frame
# the tan dock reads as `eel` and shore plants as `green` (heavy false
# positives). See docs/fishing_minigame.md.
FISH_HSV: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    # Green Fish: in-game fill is H76 S79 V255 — the S floor must be LOW (was
    # 100, which excluded the fish entirely) and the hue capped BELOW the cast
    # bar's cyan top-edge (H~98) so the fish doesn't merge into it. Verified on
    # the capture frames (one clean ~17x17 blob per frame).
    "green": ((70, 50, 130), (92, 255, 255)),
    "eel": ((13, 60, 120), (28, 255, 255)),       # Eel (yellow; sprite H18-24)
    "squid": ((132, 60, 30), (150, 255, 170)),    # Squid (purple, DARK V~60)
    # Whale (blue) overlaps the blue bar (H~109) — separated by saturation (bar
    # S~186, whale sprite S~79). Provisional until a whale is seen live.
    "whale": ((100, 40, 110), (118, 130, 255)),
}
# Fish are roughly-square SOLID sprites; the warm-hued false positives (the
# tan dock / palm / sandcastle edges that read as 'eel') are thin and wide,
# and the score text glyphs are small. Keep only square-ish, filled blobs.
FISH_ASPECT_RANGE = (0.45, 2.3)   # width/height
FISH_MIN_FILL = 0.35              # contourArea / bbox area
FISH_MIN_AREA = 70
# Red wraps the hue circle; Megalodon (red behemoth, sprite H6-10 S~81 V255).
MEGALODON_HSV_LOW = ((3, 90, 150), (12, 255, 255))
MEGALODON_HSV_HIGH = ((172, 90, 150), (179, 255, 255))
# Mines (MGfsh5) render as a grey/low-saturation spiky ball with a red core
# (in-game body H~9-19 S~30-45 V~86-130). Key off the low-saturation grey
# body; the blue bar (S~186) and the colour fish (S>100) don't match it.
# Spiky shape / template is the fallback if colour proves unreliable live.
MINE_HSV = ((0, 0, 45), (179, 62, 150))

# A detected blob must cover at least this many mask pixels to count as a
# target (filters speckle). Scale with capture resolution during calibration.
MIN_BLOB_AREA = 40

# --- Cast bar (the play track) ------------------------------------------
# The minigame is a horizontal blue bar drawn over the world (player-
# anchored), with fish/mines positioned along it; the lure is cast rightward
# to a position on the bar. Detection MUST be confined to the bar or world
# scenery floods the colour masks. The bar is a solid, high-saturation blue
# strip — distinct from the lighter, less-saturated sky/water.
BAR_HSV = ((102, 150, 90), (118, 255, 205))
BAR_MIN_WIDTH = 250          # the track spans a wide strip
BAR_MIN_ASPECT = 6.0         # much wider than tall
# Fish/mines sit ON the bar but extend above/below its thin saturated core,
# so the detection window is the bar's x-extent and its y +/- this pad.
BAR_VPAD = 20


def _to_hsv(frame: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def _mask(hsv: np.ndarray, low, high) -> np.ndarray:
    return cv2.inRange(hsv, np.array(low), np.array(high))


def _blob_centroids(mask: np.ndarray, min_area: int = MIN_BLOB_AREA,
                    aspect_range: tuple[float, float] | None = None,
                    min_fill: float = 0.0) -> list[tuple[int, int]]:
    """Centroids (x, y) of mask blobs above `min_area`, largest first.

    Optional shape gates reject non-target blobs: `aspect_range` (width/height)
    drops thin wide edges (scenery) and tall slivers; `min_fill`
    (contourArea / bbox area) drops sparse/outline blobs. Used by find_fish to
    keep the square solid fish sprites and discard the tan-scenery edges and
    score-text glyphs that share fish hues (#63)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[tuple[int, int, float]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        if aspect_range is not None and not (aspect_range[0] <= w / h <= aspect_range[1]):
            continue
        if area / (w * h) < min_fill:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        out.append((int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]), area))
    out.sort(key=lambda t: -t[2])
    return [(x, y) for x, y, _ in out]


def find_cast_bar(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Locate the cast bar (the blue play track) as (x, y, w, h), or None.

    The bar is a solid, high-saturation blue horizontal strip; the sky/water
    are lighter / less saturated, so a saturation floor + a wide-aspect filter
    isolate it. Used to confine fish/mine detection to the track (the world
    scenery shares the fish hues otherwise). Only the bar's saturated core is
    matched; callers pad vertically (BAR_VPAD) for the fish that sit on it."""
    hsv = _to_hsv(frame)
    mask = _mask(hsv, *BAR_HSV)
    # bridge the gaps the fish/mines punch in the strip so it stays one contour
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 25), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_w = None, 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= BAR_MIN_WIDTH and w / max(1, h) >= BAR_MIN_ASPECT and w > best_w:
            best, best_w = (x, y, w, h), w
    return best


def _restrict(mask: np.ndarray, bar: tuple[int, int, int, int] | None,
              vpad: int = BAR_VPAD) -> np.ndarray:
    """Zero the mask outside the cast bar's x-extent and y +/- vpad, so only
    targets on the track survive. No-op when bar is None (whole-frame)."""
    if bar is None:
        return mask
    x, y, w, h = bar
    out = np.zeros_like(mask)
    y0, y1 = max(0, y - vpad), min(mask.shape[0], y + h + vpad)
    x0, x1 = max(0, x), min(mask.shape[1], x + w)
    out[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return out


def find_fish(frame: np.ndarray, min_area: int = FISH_MIN_AREA,
              bar: tuple[int, int, int, int] | None = None,
              include_megalodon: bool = False) -> list[dict]:
    """All detected fish as dicts {x, y, kind}. Kind is one of FISH_HSV keys.
    Positions are play-region-relative. When `bar` (from find_cast_bar) is
    given, detection is confined to the track. A square+fill shape gate keeps
    the fish sprites and rejects the tan-scenery edges / score-text glyphs that
    share the warm hues (#63). Empty list when nothing matches.

    Megalodon detection is OFF by default — its red hue is shared by the mine
    cores and the (always-present) red bobber, so routine detection is just
    noise; it's a rare trophy. Enable include_megalodon once a real one can be
    distinguished (e.g. by size)."""
    hsv = _to_hsv(frame)
    fish: list[dict] = []
    for kind, (low, high) in FISH_HSV.items():
        mask = _restrict(_mask(hsv, low, high), bar)
        for x, y in _blob_centroids(mask, min_area, FISH_ASPECT_RANGE, FISH_MIN_FILL):
            fish.append({"x": x, "y": y, "kind": kind})
    if include_megalodon:
        meg = cv2.bitwise_or(_mask(hsv, *MEGALODON_HSV_LOW), _mask(hsv, *MEGALODON_HSV_HIGH))
        for x, y in _blob_centroids(_restrict(meg, bar), min_area,
                                    FISH_ASPECT_RANGE, FISH_MIN_FILL):
            fish.append({"x": x, "y": y, "kind": "megalodon"})
    return fish


def find_mines(frame: np.ndarray, min_area: int = MIN_BLOB_AREA,
               bar: tuple[int, int, int, int] | None = None) -> list[dict]:
    """Detected mines as dicts {x, y}. Mines never end the game if you land
    on a fish (wiki), but landing on a mine-only spot does — so the bot
    avoids casting where only a mine sits. Confined to the bar when given."""
    hsv = _to_hsv(frame)
    return [{"x": x, "y": y}
            for x, y in _blob_centroids(_restrict(_mask(hsv, *MINE_HSV), bar), min_area)]


def kind_at(frame: np.ndarray, x: int, y: int, radius: int = 12) -> str | None:
    """The fish kind whose blob is closest to (x, y) within `radius` px, or
    None. Used post-cast to classify what the lure landed on (-> points)."""
    best: tuple[float, str] | None = None
    for f in find_fish(frame):
        d = ((f["x"] - x) ** 2 + (f["y"] - y) ** 2) ** 0.5
        if d <= radius and (best is None or d < best[0]):
            best = (d, f["kind"])
    return best[1] if best else None


def find_lure(frame: np.ndarray, threshold: float = 0.7) -> tuple[int, int] | None:
    """Locate the lure (MGlure) via template match — where the cast landed.
    None until assets/lure.png is captured (fishing-capture). Placeholder:
    landing detection needs this template before outcomes can be logged."""
    path = ASSETS / "lure.png"
    if not path.exists():
        return None
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    template = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if template is None:
        return None
    (cx, cy), val, _scale = match_multiscale_center(bgr, template)
    return (cx, cy) if val >= threshold else None


# "PLAY GAME" entry prompt — the shared Idleon minigame button (same sprite
# as catching/mining; reused with catching's mask). Matched by background-
# invariant masked ZNCC (play_button_mask.png covers the rigid grey button,
# excluding the count-badge corner) so it holds up over the fishing biome;
# falls back to unmasked CCOEFF if the mask is absent. Thresholds mirror
# catching (0.6 masked / 0.75 unmasked).
PLAY_BUTTON_MATCH_THRESHOLD = 0.6
PLAY_BUTTON_UNMASKED_FALLBACK_THRESHOLD = 0.75


def find_play_button(frame: np.ndarray) -> tuple[int, int] | None:
    """Locate the 'PLAY GAME' entry prompt; return its (x, y) centre or None.
    Search the FULL window — the prompt is anchored above the player, not in
    the cast-bar play region. Masked ZNCC (play_button_mask.png) is the primary
    path for background invariance; unmasked CCOEFF is the fallback."""
    path = ASSETS / "play_button.png"
    if not path.exists():
        return None
    template = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if template is None:
        return None
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if bgr.shape[0] < template.shape[0] or bgr.shape[1] < template.shape[1]:
        return None
    mask_path = ASSETS / "play_button_mask.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
    if mask is not None:
        center, conf, _ = match_multiscale_zncc_center(bgr, template, mask)
        if center is None or conf < PLAY_BUTTON_MATCH_THRESHOLD:
            return None
        return center
    center, val, _ = match_multiscale_center(bgr, template)
    if center is None or val < PLAY_BUTTON_UNMASKED_FALLBACK_THRESHOLD:
        return None
    return center


def find_game_over(frame: np.ndarray, threshold: float = 0.7) -> tuple[bool, float]:
    """Detect the end-of-minigame screen via template match. (False, 0.0)
    until assets/game_over.png is captured. Whether a mine-landing ends the
    game is unconfirmed (wiki is ambiguous) — see docs/fishing_minigame.md;
    the no-fish timeout in main is the fallback bail."""
    path = ASSETS / "game_over.png"
    if not path.exists():
        return False, 0.0
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    template = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if template is None:
        return False, 0.0
    _, val, _scale = match_multiscale_center(bgr, template)
    return val >= threshold, val


def scene_fish_count(frame: np.ndarray) -> int:
    """Number of fish blobs detected — a cheap 'is the minigame active'
    proxy (the water is full of fish during play, empty otherwise), the
    fishing analogue of chopping's bar_pixel_count."""
    return len(find_fish(frame))
