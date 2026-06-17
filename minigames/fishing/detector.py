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

from common.templates import match_multiscale_center

ASSETS = Path(__file__).parent / "assets"

# OpenCV HSV: H in [0,179], S/V in [0,255]. Red wraps -> two ranges.
# PLACEHOLDER ranges — calibrate against real frames (see module note).
FISH_HSV: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "green": ((40, 80, 80), (80, 255, 255)),     # Green Fish
    "eel": ((20, 90, 90), (35, 255, 255)),       # Eel (yellow-ish)
    "squid": ((125, 70, 70), (155, 255, 255)),   # Squid (purple)
    "whale": ((90, 80, 80), (120, 255, 255)),    # Whale (blue)
}
# Red wraps the hue circle; Megalodon (red behemoth) + some mines.
MEGALODON_HSV_LOW = ((0, 120, 90), (10, 255, 255))
MEGALODON_HSV_HIGH = ((170, 120, 90), (179, 255, 255))
# Mines (MGfsh5) — dark/low-value spiky sprites. Placeholder: low value,
# low saturation. Calibrate; mines may instead key off the sprite via a
# template if colour proves unreliable.
MINE_HSV = ((0, 0, 0), (179, 80, 70))

# A detected blob must cover at least this many mask pixels to count as a
# target (filters speckle). Scale with capture resolution during calibration.
MIN_BLOB_AREA = 40


def _to_hsv(frame: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


def _mask(hsv: np.ndarray, low, high) -> np.ndarray:
    return cv2.inRange(hsv, np.array(low), np.array(high))


def _blob_centroids(mask: np.ndarray, min_area: int = MIN_BLOB_AREA) -> list[tuple[int, int]]:
    """Centroids (x, y) of mask blobs above `min_area`, largest first."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[tuple[int, int, float]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        out.append((int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]), area))
    out.sort(key=lambda t: -t[2])
    return [(x, y) for x, y, _ in out]


def find_fish(frame: np.ndarray, min_area: int = MIN_BLOB_AREA) -> list[dict]:
    """All detected fish as dicts {x, y, kind}. Kind is one of FISH_HSV
    keys or 'megalodon'. Positions are play-region-relative. Empty list
    when nothing matches (or, pre-calibration, when the ranges are off)."""
    hsv = _to_hsv(frame)
    fish: list[dict] = []
    for kind, (low, high) in FISH_HSV.items():
        for x, y in _blob_centroids(_mask(hsv, low, high), min_area):
            fish.append({"x": x, "y": y, "kind": kind})
    meg = cv2.bitwise_or(_mask(hsv, *MEGALODON_HSV_LOW), _mask(hsv, *MEGALODON_HSV_HIGH))
    for x, y in _blob_centroids(meg, min_area):
        fish.append({"x": x, "y": y, "kind": "megalodon"})
    return fish


def find_mines(frame: np.ndarray, min_area: int = MIN_BLOB_AREA) -> list[dict]:
    """Detected mines as dicts {x, y}. Mines never end the game if you land
    on a fish (wiki), but landing on a mine-only spot does — so the bot
    avoids casting where only a mine sits."""
    hsv = _to_hsv(frame)
    return [{"x": x, "y": y} for x, y in _blob_centroids(_mask(hsv, *MINE_HSV), min_area)]


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


def find_play_button(frame: np.ndarray, threshold: float = 0.7) -> tuple[int, int] | None:
    """Locate the 'play minigame' prompt (template). In-world prompts move
    with the player, so this is matched visually, not coordinate-cached
    (CLAUDE.md). None until assets/play_button.png exists."""
    path = ASSETS / "play_button.png"
    if not path.exists():
        return None
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    template = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if template is None:
        return None
    (cx, cy), val, _scale = match_multiscale_center(bgr, template)
    return (cx, cy) if val >= threshold else None


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
