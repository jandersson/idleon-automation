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
    # Eel is NOT here — it's a yellow CURLED fish whose colour (H12-24, S~122)
    # is indistinguishable from the tan dock (S~121), so HSV either misses it or
    # floods 68% of frames with dock false positives (#63). It's matched by its
    # distinctive curled SHAPE instead (find_eel, assets/eel.png) — template
    # match 1.00 on the eel vs <=0.63 across 263 no-eel frames.
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
# false positives, cross-biome (turquoise beach) green at 0.88. Whale has no
# template yet (never captured, #69) so it stays HSV-only; eel keeps its own
# curled-shape template (find_eel). Sprite library: assets/fish_<kind>_<n>.png +
# a _mask.png isolating the body (captured/validated through the live pipeline).
SPRITE_KINDS = ("green", "squid")
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
    for kind, (low, high) in FISH_HSV.items():   # green, squid, whale (square sprites)
        mask = _restrict(_mask(hsv, low, high), bar)
        for x, y in _blob_centroids(mask, min_area, FISH_ASPECT_RANGE, FISH_MIN_FILL):
            fish.append({"x": x, "y": y, "kind": kind})
    # Eel: matched by its curled SHAPE (its colour can't be told from the dock,
    # #63). Skip a match inside a mine as a safety against a chance template hit.
    eel = find_eel(frame, bar=bar)
    if eel is not None and not _in_a_mine(eel[0], eel[1], mines):
        fish.append({"x": eel[0], "y": eel[1], "kind": "eel"})
    if include_megalodon:
        meg = cv2.bitwise_or(_mask(hsv, *MEGALODON_HSV_LOW), _mask(hsv, *MEGALODON_HSV_HIGH))
        for x, y in _blob_centroids(_restrict(meg, bar), min_area,
                                    FISH_ASPECT_RANGE, FISH_MIN_FILL):
            if _in_a_mine(x, y, mines):
                continue          # mine's red core, not a megalodon
            fish.append({"x": x, "y": y, "kind": "megalodon"})
    # Background-invariant sprite detections (masked ZNCC, #75) merged ADDITIVELY:
    # a fish counts if EITHER the HSV gate above OR the sprite match fires. A
    # sprite det suppresses an HSV det only as a SAME-KIND duplicate (the redundant
    # green/squid blob at the same spot) — never a different kind. The dedup MUST
    # be kind-aware: a green/squid sprite must not delete a converged eel(2)/whale
    # (5)/megalodon sitting within DEDUP_PX, which have no sprite template and
    # would otherwise vanish from choose_target in exactly the converged-cluster
    # regime where the high-value fish matters most (#75 review). So HSV still
    # covers whale/eel/megalodon and the ~1% of green/squid a pose misses, while
    # the sprite match adds the green/squid HSV drops in a new biome.
    sprites = find_fish_sprites(frame, bar)
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


# The eel is a yellow CURLED fish; its colour (H12-24, S~122) is the tan dock's
# (S~121), so HSV can't find it — but the curl is a distinctive SHAPE, matched by
# template (assets/eel.png, cropped live from a streak-3 run). Threshold 0.75
# sits above the worst no-eel frame (0.63 across 263) and below the eel (0.95-1.0)
# (#63). One template/pose so far — widen with more eel crops if it under-matches.
EEL_MATCH_THRESHOLD = 0.75


def find_eel(frame: np.ndarray, threshold: float = EEL_MATCH_THRESHOLD,
             bar: tuple[int, int, int, int] | None = None) -> tuple[int, int] | None:
    """Locate the eel by its curled-shape template(s); (x, y) of the best match
    at/above `threshold`, else None. `bar` is accepted for a consistent signature
    but the match runs on the whole crop.

    Matches against ALL ``assets/eel*.png`` poses and keeps the best. The eel
    curls into several shapes, and a SINGLE pose under-matches the others at
    ~0.63 — right at the no-eel ceiling, so there's no threshold that separates
    them and most eels go undetected (#63). Drop additional clean eel crops
    (``eel_<tag>.png``) from a ``--save-frames`` run to widen pose coverage."""
    templates = sorted(ASSETS.glob("eel*.png"))
    if not templates:
        return None
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    best_val, best_xy = -1.0, None
    for path in templates:
        template = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if template is None:
            continue
        if bgr.shape[0] < template.shape[0] or bgr.shape[1] < template.shape[1]:
            continue
        (cx, cy), val, _scale = match_multiscale_center(bgr, template)
        if val > best_val:
            best_val, best_xy = val, (cx, cy)
    return best_xy if best_val >= threshold else None


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
# NOT at any crop edge. Measured live (run 10, 960x572): with the cast bar at
# full-window (434,233), the tube fills within x[386,412], y[170,252], i.e. an
# offset of bar_x-48..bar_x-22 and bar_y-63..bar_y+19. So it's read from the FULL
# window anchored to the detected cast bar, with margin (offsets below). The red
# fill HEIGHT is the charge level; the bobber/scenery red sits elsewhere (on/right
# of the bar), outside this left-of-bar window. See docs/fishing_minigame.md.
CHARGE_BAR_DX0 = -50   # search window, relative to the cast bar's left edge (x)
CHARGE_BAR_DX1 = -22
CHARGE_BAR_DY0 = -64   # relative to the cast bar's top (y); thermometer rises above it
CHARGE_BAR_DY1 = 18    # tuned on run-10 frames: empty->0, full->56, clean ramp


def find_charge_fill(full_frame: np.ndarray,
                     cast_bar_full: tuple[int, int, int, int] | None) -> int:
    """Red-fill height (px) of the vertical charge thermometer — the LIVE
    cast-power signal during the hold. `full_frame` is the whole window;
    `cast_bar_full` is the cast bar's (x, y, w, h) in FULL-window coords (the
    anchor). Returns 0 when empty or the anchor is missing. The closed-loop cast
    polls this and releases at a target fill."""
    if cast_bar_full is None:
        return 0
    bx, by, _bw, _bh = cast_bar_full
    h, w = full_frame.shape[:2]
    x0, x1 = max(0, bx + CHARGE_BAR_DX0), max(0, min(w, bx + CHARGE_BAR_DX1))
    y0, y1 = max(0, by + CHARGE_BAR_DY0), max(0, min(h, by + CHARGE_BAR_DY1))
    if x1 <= x0 or y1 <= y0:
        return 0
    hsv = _to_hsv(full_frame[y0:y1, x0:x1])
    red = cv2.bitwise_or(_mask(hsv, *CHARGE_RED_LOW), _mask(hsv, *CHARGE_RED_HIGH))
    return int(np.count_nonzero((red > 0).any(axis=1)))   # rows with red = fill height


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
