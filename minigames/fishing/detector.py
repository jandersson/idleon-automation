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

from common.templates import (
    match_multiscale_center, match_multiscale_zncc_center, masked_match_confidence,
)

ASSETS = Path(__file__).parent / "assets"

# OpenCV HSV: H in [0,179], S/V in [0,255]. Red wraps -> two ranges.
# Calibrated 2026-06-17 from the wiki minigame sprites (MGgreenfish/yellow/
# purple/blue/red, MGfsh5 mine) + a live observe session (960x572). `green`
# is verified in-game (H77-82); the rest are sprite-derived and still want a
# live check when they appear (the streak gates eel@3 / squid@6 / whale@13).
# Detection is restricted to the cast bar (find_cast_bar) — over the raw frame
# the tan dock reads as `eel` and shore plants as `green` (heavy false
# positives). See docs/fishing_minigame.md.
#
# RETIRED for active detection: green/squid (and eel) are masked-ZNCC SPRITES now
# (biome-invariant, #75/#77) — find_fish SKIPS any kind in SPRITE_KINDS here, so
# only `whale` is detected via HSV (no sprite captured yet, #69). The green/squid
# ranges below are retained for reference/tests + as the eventual whale-sprite
# cross-check; the green H88 cap is now historical (the sprite path made it moot).
FISH_HSV: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    # Green Fish: in-game fill is H76 S79 V255 — the S floor must be LOW (was
    # 100, which excluded the fish entirely). The hue is capped at 88: the
    # original 92 was "below the cast bar's cyan edge", but a BRIGHT BEACH biome
    # (blue sky + TURQUOISE water, vs the calibration area's orange sunset) puts
    # the water at H90-100, which flooded the green mask inside the bar window and
    # buried the fish in a non-square blob -> find_fish returned nothing and the
    # bot sat stuck (#72). The seafoam-green fish there reads <=88, the turquoise
    # water >=90, so 88 separates them with a ~2-hue margin. Validated zero-
    # regression on 215 old-area frames (green detections 179 = 179) and recovers
    # the new-area fish. One clean ~17x17 blob per frame.
    "green": ((70, 50, 130), (88, 255, 255)),
    # Eel is NOT here — its colour (H12-24, S~122) is indistinguishable from the
    # tan dock (S~121), so HSV can't find it (#63). It's a masked-ZNCC SPRITE kind
    # now (fish_eel_*.png + GrabCut body masks, in SPRITE_KINDS / find_fish_sprites,
    # #77) — the curl is matched by its pixels with the warm background masked out.
    # Squid (purple, DARK V~60). Validated live on the first real squid
    # (cast09/botrun_120520): detected in 5/6 frames, no false positives (#63).
    "squid": ((132, 60, 30), (150, 255, 170)),
    # Whale (blue) overlaps the blue bar (H~109) — separated by saturation (bar
    # S~186, whale sprite S~79). Provisional until a whale is seen live.
    "whale": ((100, 40, 110), (118, 130, 255)),
}

# Background-invariant sprite detection (#75). The minigame is an overlay over a
# LIVE, varying world, so absolute-HSV gates break per biome (tan dock -> 'eel'
# #63; turquoise water floods 'green' #72). The fish SPRITES are identical pixels
# regardless of biome, so match the sprite (masked ZNCC: mean-subtracted NCC over
# the sprite's own pixels, the world behind it dropped) — `green`/`squid` template
# from one biome scores ~0.9 on the SAME fish in any other, vs ~0.3 on background.
# Validated: 99-100% of HSV detections recovered, +8 HSV-missed fish found, 0 true
# false positives, cross-biome (turquoise beach) green at 0.88. The EEL is here
# too (#77): its colour == the dock, but a GrabCut body mask lets masked ZNCC
# match the curl's pixels — fixing the occlusion under-match (0.63 unmasked ->
# ~0.97 masked). Whale has no template yet (never captured, #69) so it stays
# HSV-only. Sprite library: assets/fish_<kind>_<n>.png + a _mask.png isolating the
# body (green/squid masks from HSV; eel masks from GrabCut, its colour==background).
SPRITE_KINDS = ("green", "squid", "eel")
SPRITE_SCALES = (0.9, 1.0, 1.1)    # fish size is window-relative; small scale slack
SPRITE_CCORR_FLOOR = 0.55          # masked-CCORR localization candidate floor
SPRITE_ZNCC_THRESHOLD = 0.55       # accept gate; true fish >=0.59 (incl. occluded), background <~0.35
SPRITE_DEDUP_PX = 12               # merge peaks/dets within this x (one fish)
SPRITE_MAX_PEAKS = 8               # per template-scale, cap the NMS loop
# Fish are roughly-square SOLID sprites; the warm-hued false positives (the
# tan dock / palm / sandcastle edges that read as 'eel') are thin and wide,
# and the score text glyphs are small. Keep only square-ish, filled blobs.
FISH_ASPECT_RANGE = (0.45, 2.3)   # width/height
FISH_MIN_FILL = 0.35              # contourArea / bbox area
FISH_MIN_AREA = 70
# Red wraps the hue circle; Megalodon (red behemoth, sprite H6-10 S~81 V255).
MEGALODON_HSV_LOW = ((3, 90, 150), (12, 255, 255))
MEGALODON_HSV_HIGH = ((172, 90, 150), (179, 255, 255))
# Mines render as a SPIKY ball whose SPIKES are RED (the core is orange). Key on
# the RED spikes, NOT orange: the bar overlays an orange sunset sky (H~15-25),
# so an orange mask floods the whole frame. Red (H<=10 / >=168) is just the mine
# spikes + the red/white bobber. A mine is a compact ball ~24-52px wide (a fish
# is ~17px; the bobber is round). The orange core inside reads as an 'eel'
# (#63), so find_fish drops fish blobs inside a detected mine. Red wraps -> two
# ranges.
MINE_RED_LOW = ((0, 120, 90), (10, 255, 255))
MINE_RED_HIGH = ((168, 120, 90), (179, 255, 255))
MINE_MIN_AREA = 110
MINE_MIN_WIDTH = 22      # a mine ball spans ~24-52px; a fish ~17px
MINE_MAX_WIDTH = 60      # reject merged warm scenery (was detecting 137px-wide blobs)
MINE_ASPECT = (0.55, 1.9)   # roughly round ball

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
# Corner cut-off (game bug): standing in the RIGHT corner, the board doesn't
# fully render — the cast bar is CLIPPED at the screen's right edge (observed
# 2026-06-18: a 230px bar pinned to x=window_w, below BAR_MIN_WIDTH -> find_cast_bar
# returned None -> the bot went blind). A bar whose right edge touches the window
# edge is accepted at a reduced floor so the VISIBLE (left) portion can still be
# played — the cast origin, charge thermometer, and score all sit at the bar's
# left, which is intact in the right corner. A LEFT-clipped bar is NOT accepted at
# the reduced floor: its left edge (the origin/charge/score anchor) is the cut-off
# side, so a short left-clipped bar would mis-anchor everything (#58) — better to
# return None and not play.
BAR_CLIP_MARGIN = 4          # bar right edge within this many px of the window = clipped
BAR_MIN_WIDTH_CLIPPED = 200  # reduced floor for a right-clipped bar (still substantial)
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
    matched; callers pad vertically (BAR_VPAD) for the fish that sit on it.

    A bar CLIPPED at the window's RIGHT edge (the corner cut-off bug) is accepted
    at the reduced BAR_MIN_WIDTH_CLIPPED floor so the visible left portion can be
    played; a left-clipped or mid-screen narrow strip still needs the full
    BAR_MIN_WIDTH (a left-clip would cut off the origin/charge/score anchor)."""
    hsv = _to_hsv(frame)
    win_w = frame.shape[1]
    mask = _mask(hsv, *BAR_HSV)
    # bridge the gaps the fish/mines punch in the strip so it stays one contour
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 25), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_w = None, 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w / max(1, h) < BAR_MIN_ASPECT:
            continue
        clipped_right = (x + w) >= win_w - BAR_CLIP_MARGIN
        min_w = BAR_MIN_WIDTH_CLIPPED if clipped_right else BAR_MIN_WIDTH
        if w >= min_w and w > best_w:
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
              include_megalodon: bool = False,
              mines: list[dict] | None = None) -> list[dict]:
    """All detected fish as dicts {x, y, kind}. Kind is one of FISH_HSV keys.
    Positions are play-region-relative. When `bar` (from find_cast_bar) is
    given, detection is confined to the track. A square+fill shape gate keeps
    the fish sprites and rejects the tan-scenery edges / score-text glyphs that
    share the warm hues (#63). Empty list when nothing matches.

    A mine's bright ORANGE core reads as an 'eel' (#63). Pass `mines`
    (find_mines output) — or leave None to detect them here — and any fish blob
    centred inside a mine is dropped, so mine cores aren't mislabelled fish.

    Megalodon detection is OFF by default — its red hue is shared by the mine
    cores and the (always-present) red bobber, so routine detection is just
    noise; it's a rare trophy. Enable include_megalodon once a real one can be
    distinguished (e.g. by size)."""
    if mines is None:
        mines = find_mines(frame, bar=bar)
    hsv = _to_hsv(frame)
    fish: list[dict] = []
    for kind, (low, high) in FISH_HSV.items():
        # HSV is RETIRED for any kind that has a masked-ZNCC sprite (#75): green/
        # squid/eel are sprite-only now (biome-invariant — they match the sprite's
        # pixels, not an absolute hue), so the HSV loop runs ONLY for kinds without
        # a sprite — currently just `whale` (uncaptured, #69). This is the last
        # biome-dependent fish path; it goes away when a whale sprite is captured.
        if kind in SPRITE_KINDS:
            continue
        mask = _restrict(_mask(hsv, low, high), bar)
        for x, y in _blob_centroids(mask, min_area, FISH_ASPECT_RANGE, FISH_MIN_FILL):
            fish.append({"x": x, "y": y, "kind": kind})
    if include_megalodon:
        meg = cv2.bitwise_or(_mask(hsv, *MEGALODON_HSV_LOW), _mask(hsv, *MEGALODON_HSV_HIGH))
        for x, y in _blob_centroids(_restrict(meg, bar), min_area,
                                    FISH_ASPECT_RANGE, FISH_MIN_FILL):
            if _in_a_mine(x, y, mines):
                continue          # mine's red core, not a megalodon
            fish.append({"x": x, "y": y, "kind": "megalodon"})
    # Background-invariant sprite detections (masked ZNCC) for green/squid/eel
    # (#75/#77) — the PRIMARY (and now ONLY) path for those kinds; HSV above is
    # retired for them, so biome can't affect their detection. Merged with the
    # remaining HSV kinds (whale, megalodon): a sprite det suppresses an HSV det
    # only as a SAME-KIND duplicate — but green/squid/eel are no longer in `fish`,
    # so in practice the merge just appends the HSV whale/megalodon (different
    # kinds, never deduped). The kind-aware dedup is kept defensively: were a
    # green/squid ever re-added to HSV, it must not delete a converged
    # whale(5)/megalodon within DEDUP_PX (the #75-review regression).
    sprites = find_fish_sprites(frame, bar)
    # Preserve the eel-in-mine safety from the old curled-template path: drop an
    # eel sprite det centred inside a mine (a chance match on a mine core).
    sprites = [s for s in sprites
               if not (s["kind"] == "eel" and _in_a_mine(s["x"], s["y"], mines))]
    if sprites:
        merged = list(sprites)
        for f in fish:
            if all(f["kind"] != s["kind"] or abs(f["x"] - s["x"]) > SPRITE_DEDUP_PX
                   for s in sprites):
                merged.append(f)
        return merged
    return fish


def find_mines(frame: np.ndarray,
               bar: tuple[int, int, int, int] | None = None) -> list[dict]:
    """Detected mines as dicts {x, y, bbox=(x,y,w,h)}. A mine is a large
    saturated red/orange blob (the spiky ball + core), distinct from the ~17px
    fish and the dull tan scenery. Confined to the bar when given. The bbox lets
    find_fish drop a mine's orange core, which else reads as an 'eel' (#63).

    Mines never end the game if you land on a fish (wiki), but a mine-only spot
    does — so the bot avoids casting where only a mine sits."""
    hsv = _to_hsv(frame)
    red = cv2.bitwise_or(_mask(hsv, *MINE_RED_LOW), _mask(hsv, *MINE_RED_HIGH))
    red = _restrict(red, bar)
    # bridge the gaps between the spikes so the ball is one contour
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[dict] = []
    for c in contours:
        if cv2.contourArea(c) < MINE_MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (MINE_MIN_WIDTH <= w <= MINE_MAX_WIDTH):
            continue
        if not (MINE_ASPECT[0] <= w / max(1, h) <= MINE_ASPECT[1]):
            continue
        out.append({"x": x + w // 2, "y": y + h // 2, "bbox": (x, y, w, h)})
    return out


def _in_a_mine(x: int, y: int, mines: list[dict], pad: int = 5) -> bool:
    """Whether (x, y) falls inside any detected mine's bbox (padded). Used to
    drop a mine's orange core from fish detection (it else reads as an eel)."""
    for m in mines:
        mx, my, mw, mh = m["bbox"]
        if mx - pad <= x <= mx + mw + pad and my - pad <= y <= my + mh + pad:
            return True
    return False


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


def find_eel(frame: np.ndarray,
             bar: tuple[int, int, int, int] | None = None) -> tuple[int, int] | None:
    """(x, y) of the best eel match, or None — a thin wrapper over the masked-ZNCC
    sprite path (find_fish_sprites). The eel MIGRATED from an unmasked curled-shape
    template to the same masked-ZNCC framework as green/squid (#77): its colour
    can't be told from the warm dock/sky, but the SPRITE can be matched with a
    GrabCut body mask, which fixes the occlusion under-match (the squid-overlapped
    eel scored 0.63 unmasked / undetected, ~0.97 masked) and unifies the detector.

    Kept as the eel-presence check for main.py's eel-absence catch test. The eel is
    distinguished from the structurally-similar green by find_fish_sprites'
    highest-confidence-per-location dedup (eel ~0.98 beats green-on-eel ~0.82).
    Coords are play-region-relative."""
    eels = [d for d in find_fish_sprites(frame, bar) if d["kind"] == "eel"]
    if not eels:
        return None
    best = max(eels, key=lambda d: d["conf"])
    return (best["x"], best["y"])


_SPRITE_CACHE: dict[str, list[tuple[np.ndarray, np.ndarray]]] | None = None


def _load_sprite_templates() -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Load + cache the per-kind (sprite, mask) template pairs from
    assets/fish_<kind>_<n>.png / _mask.png. Cached at module level — find_fish
    runs every poll, so this reads disk once."""
    global _SPRITE_CACHE
    if _SPRITE_CACHE is not None:
        return _SPRITE_CACHE
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for kind in SPRITE_KINDS:
        pairs = []
        for p in sorted(ASSETS.glob(f"fish_{kind}_*.png")):
            if p.name.endswith("_mask.png"):
                continue
            t = cv2.imread(str(p), cv2.IMREAD_COLOR)
            m = cv2.imread(str(p.with_name(p.stem + "_mask.png")), cv2.IMREAD_GRAYSCALE)
            if t is not None and m is not None and t.shape[:2] == m.shape[:2]:
                pairs.append((t, m))
        out[kind] = pairs
    _SPRITE_CACHE = out
    return out


def _match_sprite_instances(region_bgr: np.ndarray,
                            templates: list[tuple[np.ndarray, np.ndarray]],
                            scales=SPRITE_SCALES,
                            ccorr_floor=SPRITE_CCORR_FLOOR,
                            zncc_thresh=SPRITE_ZNCC_THRESHOLD,
                            max_peaks=SPRITE_MAX_PEAKS) -> list[tuple[int, int, float]]:
    """All (cx, cy, zncc) matches of any pose in `templates` within `region_bgr`.

    Two-stage so it finds MULTIPLE fish and stays background-invariant: a masked
    CCORR sweep localizes candidate peaks (cheap, but scores bright-flat regions
    high), each confirmed by masked ZNCC (mean-subtracted -> rejects flat scenery,
    masked -> ignores the world behind the sprite). Iterative max + neighbourhood
    suppression on the CCORR map yields several peaks per pose. Coords are
    region-relative; the caller offsets to play-region coords. Pure CV."""
    dets: list[tuple[int, int, float]] = []
    for tmpl, mask in templates:
        for s in scales:
            th, tw = int(round(tmpl.shape[0] * s)), int(round(tmpl.shape[1] * s))
            if th < 6 or tw < 6 or region_bgr.shape[0] < th or region_bgr.shape[1] < tw:
                continue
            t = cv2.resize(tmpl, (tw, th))
            m = cv2.resize(mask, (tw, th))
            cc = cv2.matchTemplate(region_bgr, t, cv2.TM_CCORR_NORMED, mask=m)
            cc = np.nan_to_num(cc, nan=0.0, posinf=0.0, neginf=0.0)
            for _ in range(max_peaks):
                _, mx, _, loc = cv2.minMaxLoc(cc)
                if mx < ccorr_floor:
                    break
                px, py = loc
                z = masked_match_confidence(region_bgr, tmpl, mask, (px, py), s)
                if z >= zncc_thresh:
                    dets.append((px + tw // 2, py + th // 2, z))
                cc[max(0, py - th // 2):py + th // 2 + 1,
                   max(0, px - tw // 2):px + tw // 2 + 1] = 0.0
    return dets


def find_fish_sprites(frame: np.ndarray,
                      bar: tuple[int, int, int, int] | None) -> list[dict]:
    """Background-invariant fish detection by masked-ZNCC sprite matching (#75):
    {x, y, kind, conf} dicts in play-region coords, confined to the cast-bar band.
    Matches each kind's sprite poses (assets/fish_<kind>_*.png) and keeps every
    peak that clears the ZNCC gate, deduped across kinds/poses by proximity
    (highest conf wins). Empty when `bar` is None or no templates exist.

    Unlike the HSV gates this is biome-invariant — it matches the sprite's pixels,
    not their colour against whatever world is drawn behind the overlay — so it
    holds up where absolute HSV breaks per biome (#63/#72). Merged additively by
    find_fish; whale (no template) and the eel keep their existing paths."""
    if bar is None:
        return []
    templates = _load_sprite_templates()
    if not any(templates.values()):
        return []
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    x, y, w, h = bar
    y0, y1 = max(0, y - BAR_VPAD), min(bgr.shape[0], y + h + BAR_VPAD)
    x0, x1 = max(0, x), min(bgr.shape[1], x + w)
    region = bgr[y0:y1, x0:x1]
    if region.size == 0:
        return []
    raw: list[tuple[int, int, str, float]] = []
    for kind in SPRITE_KINDS:
        for dx, dy, z in _match_sprite_instances(region, templates[kind]):
            raw.append((x0 + dx, y0 + dy, kind, z))
    raw.sort(key=lambda d: -d[3])      # highest confidence first
    out: list[dict] = []
    for fx, fy, kind, z in raw:
        if all(abs(fx - o["x"]) > SPRITE_DEDUP_PX for o in out):
            out.append({"x": fx, "y": fy, "kind": kind, "conf": round(z, 3)})
    return out


# "PLAY GAME" entry prompt — the shared Idleon minigame button (same sprite
# as catching/mining; reused with catching's mask). Matched by background-
# invariant masked ZNCC (play_button_mask.png covers the rigid grey button,
# excluding the count-badge corner) so it holds up over the fishing biome;
# falls back to unmasked CCOEFF if the mask is absent. Thresholds mirror
# catching (0.6 masked / 0.75 unmasked).
PLAY_BUTTON_MATCH_THRESHOLD = 0.6
PLAY_BUTTON_UNMASKED_FALLBACK_THRESHOLD = 0.75


# Charge (power) bar — the red vertical strip at the far-left edge that fills
# while holding and sets the cast distance. It's always in-crop and stable
# after release, so its fill height is a far more robust distance signal than
# tracking the bobber (which arcs off-crop and reels in). Measured live: fill
# height correlates ~linearly with landing distance (~4.9 px-distance per px,
# reproducible). Restricted to x < CHARGE_BAR_X_MAX so the red-white beach
# umbrella (a bit further right) doesn't leak in.
CHARGE_BAR_X_MAX = 10
CHARGE_RED_LOW = ((0, 120, 90), (10, 255, 255))
CHARGE_RED_HIGH = ((170, 120, 90), (179, 255, 255))


def find_charge_level(frame: np.ndarray) -> int:
    """Fill height (px) of the left-edge red charge bar — 0 when empty. The
    cast-power signal: maps ~linearly to landing distance, robustly (in-crop,
    stable post-release), unlike the off-crop bobber landing. The closed-loop
    cast polls this every ~25ms while charging, so the left strip is sliced off
    BEFORE the HSV convert (per-pixel, identical result, ~Nx less work).

    NOTE: this reads x<10 of whatever crop it's given. The LIVE charge meter is a
    vertical thermometer LEFT of the cast bar (find_charge_fill), not at a crop
    edge — this function only caught a post-release left bar when the player's
    position happened to put it in-crop (#58). Kept for that post-release read /
    back-compat; the closed-loop cast uses find_charge_fill."""
    hsv = _to_hsv(frame[:, :CHARGE_BAR_X_MAX])
    red = cv2.bitwise_or(_mask(hsv, *CHARGE_RED_LOW), _mask(hsv, *CHARGE_RED_HIGH))
    return int(np.count_nonzero((red > 0).any(axis=1)))   # rows with any red = fill height


# The LIVE charge meter is a vertical RED thermometer that fills bottom-up while
# the button is held, anchored just LEFT of (and extending above) the cast bar —
# NOT at any crop edge. Measured (960x572): the tube OUTLINE sits at bar_x-37 /
# bar_x-27, its INTERIOR at bar_x-36..bar_x-28, spanning bar_y-63..bar_y+19.
#
# BACKGROUND-INVARIANT read (#83). An ABSOLUTE red gate (the old V>=90 & S>=120)
# read the fill STUCK-LOW after an area move: on the bright turquoise-water area
# the fill desaturates to a pale orange (H~14, S~118, V~88) the gate misses, so
# the closed-loop never saw the fill rise and fired at uncontrolled max charge
# (~10% of casts). No fixed gate works, either — that pale fill is the SAME colour
# as the EMPTY tube under the orange-sunset calibration area (H~15, S~135, V~83),
# so an absolute threshold can't tell "washed-out fill" from "empty tube".
#
# The invariant signal is RELATIVE: the fill is REDDER (higher red-channel
# dominance, R - max(G,B)) than the background IMMEDIATELY BESIDE the tube, while
# an EMPTY tube shows the same scenery as its surroundings (no excess red). So
# per row, compare the tube interior's red-dominance to the background just
# left/right of it; a row is filled where the interior beats the adjacent
# background by CHARGE_FILL_MARGIN. This reads the fill the same whether the
# scenery behind/around it is turquoise water or orange sunset. Validated to
# within ~3px of the old reader across 176 calibration frames (full 63->66,
# partial 46->49, empty 0->3) and reads the pale-water beach frame 9->67. See
# docs/fishing_minigame.md.
CHARGE_TUBE_DX0 = -36   # tube interior, relative to the cast bar's left edge (x)
CHARGE_TUBE_DX1 = -28
CHARGE_LBG_DX0 = -46    # background reference just LEFT of the tube
CHARGE_LBG_DX1 = -40
CHARGE_RBG_DX0 = -25    # background reference just RIGHT of the tube (toward the bar)
CHARGE_RBG_DX1 = -19
CHARGE_TUBE_DY0 = -63   # tube vertical extent, relative to the cast bar's top (y)
CHARGE_TUBE_DY1 = 19
CHARGE_FILL_MARGIN = 18  # interior must beat the adjacent background's red-dominance by this


def _charge_reddom_rows(bgr: np.ndarray, bx: int, by: int, dx0: int, dx1: int,
                        h: int, w: int) -> np.ndarray | None:
    """Per-row median red-channel dominance (R - max(G,B)) of the vertical strip
    bar_x+dx0..bar_x+dx1 over the tube y-band, or None if the strip is off-frame.
    All strips share the tube y-band so their per-row arrays align for subtraction."""
    x0, x1 = max(0, bx + dx0), max(0, min(w, bx + dx1))
    y0, y1 = max(0, by + CHARGE_TUBE_DY0), max(0, min(h, by + CHARGE_TUBE_DY1))
    if x1 <= x0 or y1 <= y0:
        return None
    strip = bgr[y0:y1, x0:x1].astype(np.int16)
    reddom = strip[:, :, 2] - np.maximum(strip[:, :, 0], strip[:, :, 1])
    return np.median(reddom, axis=1)


def find_charge_fill(full_frame: np.ndarray,
                     cast_bar_full: tuple[int, int, int, int] | None) -> int:
    """Fill height (px) of the vertical charge thermometer — the LIVE cast-power
    signal during the hold. `full_frame` is the whole window; `cast_bar_full` is
    the cast bar's (x, y, w, h) in FULL-window coords (the anchor). Returns 0 when
    empty or the anchor is missing. The closed-loop cast polls this and releases
    at a target fill.

    Background-invariant (#83): a row counts as filled where the tube interior is
    more RED-DOMINANT than the background just beside the tube (the fill is redder
    than its surroundings; an empty tube isn't), so it reads the same over any
    scenery — see the module comment above."""
    if cast_bar_full is None:
        return 0
    bx, by = cast_bar_full[0], cast_bar_full[1]
    h, w = full_frame.shape[:2]
    bgr = cv2.cvtColor(full_frame, cv2.COLOR_BGRA2BGR) if full_frame.ndim == 3 and full_frame.shape[2] == 4 else full_frame
    interior = _charge_reddom_rows(bgr, bx, by, CHARGE_TUBE_DX0, CHARGE_TUBE_DX1, h, w)
    if interior is None:
        return 0
    left = _charge_reddom_rows(bgr, bx, by, CHARGE_LBG_DX0, CHARGE_LBG_DX1, h, w)
    right = _charge_reddom_rows(bgr, bx, by, CHARGE_RBG_DX0, CHARGE_RBG_DX1, h, w)
    # The truer background is the LESS-red side (avoids over-subtracting where one
    # side has warm scenery). Fall back to whichever reference is on-frame.
    if left is not None and right is not None:
        bg = np.minimum(left, right)
    elif left is not None:
        bg = left
    elif right is not None:
        bg = right
    else:
        return 0
    return int(np.count_nonzero(interior - bg > CHARGE_FILL_MARGIN))


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
