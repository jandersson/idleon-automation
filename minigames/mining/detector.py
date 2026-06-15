"""Mining detector.

Per the trace analysis (issue #1 resolution), the cart sprite is fixed
on screen while the minigame is active and the track scrolls past it
from right to left. So `find_cart` locates the cart per-frame (its
screen x depends on the player character's world position when the
minigame opens), and `find_next_terrain` scans the plank to the right
of the cart for incoming obstacles.

Obstacle types (verified visually against trace_20260515_220235 and
trace_20260516_131332):

- Pit: dark gap in the wooden plank with dark spikes. Lose-condition
  if the cart falls in. This is the only hazard the survival policy
  reacts to.
- Ore: the slam-to-score target. NOT YET CHARACTERISED — the silver-blue
  crystal piles above the plank turned out to be static parallax
  background (fixed screen x while the track scrolls), and the tan signal
  above the plank is the cave wall, so ore detection is currently disabled
  (see _scan_plank_ore). Real ore needs a scoring run + a scroll-velocity
  foreground/background filter to identify.

The plank's screen y-position varies by window size (the minigame UI
doesn't scale linearly), so we auto-detect the plank-top y per frame
by scanning for the brightest tan-hued horizontal band. Pit, ore, and
cart scans are all anchored to that y. This keeps the detector
window-size-independent.

Cart detection uses BACKGROUND-MASKED multi-template matching against a
small set of cart sprites (assets/cart_*.png) captured at different
window resolutions. Each template carries a silhouette mask (plank-tan
background excluded) and is matched via TM_CCORR_NORMED over the mask
only, so the cave/ore that scrolls behind the cart during a jump
doesn't move the score — the cart stays detected through the whole
arc, which an unmasked TM_CCOEFF_NORMED match did not (it dropped below
threshold for the launch/peak/pre-land poses). The template that scores
highest above CART_MATCH_THRESHOLD wins. A single grounded template
generalizes across jump poses; add new templates only if matching fails
at a new resolution.
"""
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from common.templates import match_multiscale_center, match_multiscale_masked
from common.score_template_ocr import make_score_reader

_HERE = Path(__file__).parent

# Score readout: the "N PTS" counter sits just BELOW-LEFT of the plank
# ("N PTS" on the left, "N BEST" on the right). It's plank-relative
# because the minigame overlay floats above the player's world position,
# so there's no fixed regions.json rectangle — the digit zone is computed
# from the detected plank each frame. Offsets derived from
# trace_20260515_220235 (plank_y=312, plank_x0=397 → the "0" digit sits
# at x~388, y~332). The right edge stops before the "PTS" label so the
# P/T/S glyphs don't get read as digits. Multi-digit scores (10+) may
# need the box widened — validate at the game.
# The number is right-anchored against the " PTS" label (its units digit
# ends ~plank_x0), so the box extends LEFT for multi-digit scores and
# stops just right of plank_x0 to exclude the "P" glyph.
# Live confirmation (2026-06-15, digit_capture/20260615_002352): at
# plank_x0=70 the units "0" occupied crop cols [plank_x0-4, plank_x0+1] —
# a clean 8x6 glyph matching 0.png. Left extent widened -24 -> -32 to give
# 3-digit headroom (scores reach ~103) without nearing the right-side
# "PTS" label; right edge kept at +2 (proven to capture the units digit
# cleanly). Still pending validation on a real 2-3 digit score.
PTS_DX = (-32, 2)    # x offset from plank_x0 spanning the score digits
PTS_DY = (20, 42)    # y offset below plank_y

# Per-game digit templates for the PTS reader (bootstrapped from gameplay
# frames; missing digits → read_score returns None for scores containing
# them, same graceful-degradation contract as the other bots).
_score_reader = make_score_reader(_HERE / "assets" / "digit_templates")

# Scans start a few px past the cart's right edge so its trailing-edge
# pixels don't read as the start of an obstacle.
SCAN_BUFFER_PX = 10

# Cart matching: try every template in assets/cart_*.png at multi-scale,
# pick the best masked match above CART_MATCH_THRESHOLD.
#
# Matching is BACKGROUND-MASKED (TM_CCORR_NORMED over the cart silhouette
# only, plank-tan pixels excluded). The cart sprite is rigid, but during a
# jump the background scrolling behind it changes from plank to cave-wall +
# ore — which dropped an unmasked TM_CCOEFF_NORMED match to ~0.75-0.79
# (below threshold) for the launch/peak/pre-land poses, leaving the bot
# blind for ~half the arc (validated on botrun_20260615_012340: cart=None
# at frames 34/36/37/41/42). Masking the background out closes that gap: a
# single grounded cart_small template generalizes to every airborne pose
# (grounded 0.99, airborne arc 0.86-1.00, see /tmp probes 2026-06-15). The
# old 0.80 threshold was for CCOEFF; masked CCORR scores higher, so the
# threshold is 0.85 — above the game-over-screen noise floor (~0.81) and
# below the worst real airborne frame (0.86). Game-over frames are anyway
# gated out upstream: _find_plank_top_y returns None there, so find_cart
# returns None before matching.
CART_MATCH_THRESHOLD = 0.85
# Scale 0.5 dropped: no real cart pose needs it (slam matches at 0.75, peak
# at 1.0), and a half-size masked template scores ~0.89 on off-column
# background noise — the launch/pre-land "ghost" at the wrong x
# (botrun_20260615_012340 frames 34/41/42 jumped to cart_x=216 at scale
# 0.5 before this; 0.6+ tracks them at the cart's true x~150).
CART_SCALES = (0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5)
# (name, template_bgr, mask) triples; mask is the cart silhouette (255) vs
# plank-tan background (0). Lazy-loaded.
_cart_templates: Optional[list[Tuple[str, np.ndarray, np.ndarray]]] = None

# Cart tracking: the cart's screen-x is fixed for a run (only its y moves
# on jumps/slams — issue #1), so after the first full detection we search
# only a narrow column around the last x (all templates + all scales, so a
# jump/slam pose change is still found in-column). This cuts the per-frame
# cost ~5x — the loop ran at ~2 FPS without it (3 templates x 8 scales x
# up-to-3 calls/frame), far too coarse for a 94 px/s scroll. A narrow miss
# falls back to a full search, so a stale prior (new attempt / different
# player position) self-heals.
#
# An earlier scale-locked variant hit ~28 FPS grounded but collapsed to
# ~5 FPS mid-jump: the locked scale missed the jump pose and fell back to
# a full search every frame (the live 2026-06-15 bot run showed this; the
# trace it was validated on had no jumps). Searching all scales in the
# (small) column trades a little steady-state FPS for staying fast through
# the jump — which is exactly when the policy needs to see the cart.
#
# Half-width 50 (was 90): the cart's x is fixed per run, so the column only
# has to absorb the ~3px click jitter + detection noise, not real motion. A
# wider column let small-scale masked matches win on off-column background
# (botrun_20260615_012340 frame 34: launch pose tracked at x=80 with M=90 vs
# ~150 with M=50). Mid-arc frames (35-41) track at the true x either way;
# the launch (34) and the dying pre-pit frame (42) are the only ones whose
# x is approximate (~30px), and the bot can't re-fire mid-jump anyway, so a
# slightly-off cart_right on those two edge frames doesn't reach a decision.
CART_TRACK_X_MARGIN = 50       # px half-width of the prior-anchored column

# Play Game button matching. The button has a per-attempt counter ("5",
# "4", ...) rendered on a wheel icon at the right edge, so the template
# isn't pixel-perfect across captures — a slightly looser threshold
# accommodates the changing digit.
PLAY_BUTTON_MATCH_THRESHOLD = 0.65
PLAY_BUTTON_SCALES = (0.5, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5)
_play_button_template: Optional[np.ndarray] = None  # lazy-loaded

# Plank-surface signature (HSV): tan/wood — H in [5,20], S>=80, V>=120.
PLANK_H_LO, PLANK_H_HI = 5, 20
PLANK_S_MIN = 80
PLANK_V_MIN = 120
PLANK_MIN_SCORE = 60  # plank-signature pixel count required to claim detection

# Plank y-search range as fraction of window height. The minigame overlay
# anchors above the player character's world position; the player is
# usually in the lower half of the world view, putting the plank in the
# upper portion. Range is generous to cover different windowed/fullscreen
# states and aspect ratios.
PLANK_Y_FRAC_MIN = 0.10
PLANK_Y_FRAC_MAX = 0.60

# Coarse plank x-search range as fraction of window width. The actual
# plank extent is detected per frame (via _find_plank_x_range) — the
# minigame overlay is a fixed-pixel-width panel that floats above the
# player's world position, so we can't hardcode its bounds as a window
# fraction. These constants only bound the search.
PLANK_X0_FRAC, PLANK_X1_FRAC = 0.0, 1.0

# Pit scan: rows just below the plank top, in the plank's body. Dark
# columns (V<PIT_V_MAX) for at least PIT_MIN_WIDTH px == pit.
PIT_SCAN_DY = (2, 7)  # rows below plank top
PIT_V_MAX = 80
PIT_MIN_WIDTH = 5

# Adjacent runs separated by this many or fewer plank-tan columns get
# merged — one physical pit often has a narrow bright spike inside that
# would otherwise split it into two false-positive runs.
MERGE_GAP_PX = 4

# Plank-column clustering gap: while finding the plank's x-extent, gaps
# wider than this split a cluster (so cave/UI tan-bright pixels far from
# the plank don't extend the detected plank). Tuned to bridge real pits
# (typically 30-80px wide) but not the cave wall on the other side of
# the play area.
PLANK_CLUSTER_GAP = 100

# Ore scan: rows ABOVE the plank top (chunks protrude up into normally
# dark cave-wall area). Bright tan-ish columns there == ore.
#
# KNOWN BROKEN — does not detect the real slam-to-score ore, and currently
# only HARMS survival (see _scan_plank_ore and docs/mining_plan.md, Run 6
# 2026-06-15). Two facts established offline against botrun_20260615_012340:
#   (1) there is no distinct "orange obstacle" — the tan signal above the
#       plank is the continuous cave wall, not a trap; the lethal hazard is
#       the pit (dark plank gap), handled by the pit scan.
#   (2) the silver-blue crystal piles are STATIC parallax background, not
#       track ore: across every grounded frame they hold fixed screen x
#       (91/196/301/406) while the foreground pit scrolls ~3px/frame. So
#       neither tan (cave wall) nor blue (background) above-plank scanning
#       finds slam-ore — which never appeared in this 0-score capture.
# Real slam-ore can only be characterised from a scoring --watch run, and
# distinguishing it will need foreground/background separation by scroll
# velocity (foreground ~93px/s, background static). Until then ore detection
# is disabled (returns []) so phantom ore can't mask pits in find_next_terrain.
ORE_SCAN_DY = (-15, -2)  # rows above plank top
ORE_V_MIN = 80
ORE_MIN_WIDTH = 5


def find_play_button(frame) -> Optional[Tuple[int, int]]:
    """Locate the in-game 'Play Game' button via multi-scale template
    matching against assets/play_button.png. Returns (center_x, center_y)
    in frame coords, or None if no match above PLAY_BUTTON_MATCH_THRESHOLD.

    Resilient to environment changes (different window size, player
    position, etc.) where the saved regions.json fractions wouldn't apply
    — the button's screen position depends on where the player character
    stands in the world."""
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    template = _load_play_button_template()
    if template is None:
        return None
    center, val, _ = match_multiscale_center(
        frame, template, scales=PLAY_BUTTON_SCALES,
    )
    if center is None or val < PLAY_BUTTON_MATCH_THRESHOLD:
        return None
    return center


def _load_play_button_template() -> Optional[np.ndarray]:
    global _play_button_template
    if _play_button_template is None:
        p = _HERE / "assets" / "play_button.png"
        if p.exists():
            _play_button_template = cv2.imread(str(p))
    return _play_button_template


def pts_score_region(
    plank_y: Optional[int],
    plank_x_range: Optional[Tuple[int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    """Box (x0, y0, x1, y1) of the PTS digit zone, computed from the
    detected plank. None if either input is missing. Pure geometry —
    testable without a frame."""
    if plank_y is None or plank_x_range is None:
        return None
    px0, _px1 = plank_x_range
    return (px0 + PTS_DX[0], plank_y + PTS_DY[0], px0 + PTS_DX[1], plank_y + PTS_DY[1])


def score_crop(
    frame,
    plank_y: Optional[int],
    plank_x_range: Optional[Tuple[int, int]],
):
    """Return the BGR PTS digit-zone crop computed from the detected
    plank, or None if the box is undefined or falls outside the frame.

    Split out from read_score so callers can grab the exact crop the OCR
    sees — the digit-template capture path (digit_capture.DigitCapturer)
    needs the same pixels, and re-deriving the geometry in two places
    would let them drift apart."""
    box = pts_score_region(plank_y, plank_x_range)
    if box is None:
        return None
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    h, w = frame.shape[:2]
    x0 = max(0, box[0]); y0 = max(0, box[1])
    x1 = min(w, box[2]); y1 = min(h, box[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1]


def read_score(
    frame,
    plank_y: Optional[int],
    plank_x_range: Optional[Tuple[int, int]],
) -> Optional[int]:
    """OCR the live PTS score from the plank-relative readout. Returns the
    integer, or None when the plank isn't located, the digit templates
    aren't bootstrapped yet, or the read fails (graceful — same contract
    as common.score_ocr.read_score)."""
    return read_score_from_crop(score_crop(frame, plank_y, plank_x_range))


def read_score_from_crop(crop) -> Optional[int]:
    """OCR a score from an already-extracted crop (see score_crop). None
    on a None crop or an unreadable digit — same graceful contract as
    read_score. Lets the main loop crop once, feed the same pixels to the
    digit capturer, and OCR them without a second crop."""
    if crop is None:
        return None
    return _score_reader(crop)


def find_cart(frame) -> Optional[Tuple[int, int]]:
    """Locate the cart sprite center, or None if no template matched above
    CART_MATCH_THRESHOLD. Thin wrapper over find_cart_detailed for callers
    / tests that only need the (x, y)."""
    res = find_cart_detailed(frame)
    return res["center"] if res is not None else None


def find_cart_detailed(frame, plank_y=None, prior=None) -> Optional[dict]:
    """Locate the cart and return {center, half_width, scale, template,
    score}, or None if no template scored above CART_MATCH_THRESHOLD.

    half_width comes free from the winning match, so callers get cart_right
    without the separate 24-match search _estimate_cart_half_width did.

    prior: a previous find_cart_detailed result. The cart's x is fixed for
    a run (issue #1), so given a prior we first search a narrow column
    around its x at its scale +/- CART_TRACK_SCALE_STEPS (fast). On a miss
    we fall back to a full-frame, all-scales search, so a stale prior (new
    attempt, different player position, or a slam-pose change) self-heals
    within a frame.

    Search is restricted to a band from the frame top down to the plank
    (the cart rises into it on a jump and sits on it otherwise)."""
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if plank_y is None:
        plank_y = _find_plank_top_y(frame)
    if plank_y is None:
        return None
    if prior is not None and prior.get("center") is not None:
        res = _match_cart(frame, plank_y, x_center=prior["center"][0])
        if res is not None:
            return res
    return _match_cart(frame, plank_y)


def find_next_terrain(frame, cart, plank_y=None, cart_right=None) -> Optional[dict]:
    """Return the nearest obstacle (pit or ore) ahead of the cart.

    Result: {"kind": "pit"|"ore", "x": int, "distance_px": int} where
    x is the left edge of the obstacle and distance_px is x - cart_right.

    plank_y and cart_right are accepted so the hot path can pass values it
    already computed this frame. Recomputing them here cost a redundant
    _find_plank_top_y plus a full 24-match cart search (via
    _estimate_cart_half_width) — the latter dominated the loop. They fall
    back to recomputation when omitted, so existing callers/tests are
    unaffected.
    """
    if cart is None:
        return None
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if plank_y is None:
        plank_y = _find_plank_top_y(frame)
    if plank_y is None:
        return None
    # cart[0] is center_x; estimate cart_right from the matched template
    # width unless the caller already has it.
    if cart_right is None:
        cart_right = cart[0] + _estimate_cart_half_width(frame, plank_y)
    scan_x = cart_right + SCAN_BUFFER_PX
    # Bound the scan to the actual plank x-extent — the minigame overlay
    # is a fixed-pixel-width panel, not full-window-wide, so cave/UI
    # beyond the plank edge can otherwise read as obstacles.
    plank_range = _find_plank_x_range(frame, plank_y)
    x_end = plank_range[1] if plank_range else None
    pits = _scan_plank_pits(frame, plank_y, x_start=scan_x, x_end=x_end)
    ores = _scan_plank_ore(frame, plank_y, x_start=scan_x, x_end=x_end)
    candidates = (
        [("pit", a, b) for a, b in pits] +
        [("ore", a, b) for a, b in ores]
    )
    if not candidates:
        return None
    nearest = min(candidates, key=lambda c: c[1])
    return {"kind": nearest[0], "x": nearest[1], "distance_px": nearest[1] - cart_right}


def _build_cart_mask(template: np.ndarray) -> np.ndarray:
    """Cart-silhouette mask (255 = cart, 0 = background) for masked matching.

    The cart templates are cropped against the plank, so the background is
    plank-tan (the same HSV signature the plank detector keys on). We mask
    OUT those pixels, leaving the cart body/rim/wheels — so the match scores
    only the sprite and ignores whatever scrolls behind it mid-jump. A
    morphological open drops isolated tan-edge specks that survive the hue
    test."""
    hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    is_plank = ((hsv[:, :, 0] >= PLANK_H_LO) & (hsv[:, :, 0] <= PLANK_H_HI) &
                (hsv[:, :, 1] >= PLANK_S_MIN) & (hsv[:, :, 2] >= PLANK_V_MIN))
    mask = (~is_plank).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _load_cart_templates() -> list[Tuple[str, np.ndarray, np.ndarray]]:
    """Lazy-load all assets/cart_*.png templates with their silhouette masks."""
    global _cart_templates
    if _cart_templates is None:
        templates = []
        for p in sorted((_HERE / "assets").glob("cart_*.png")):
            t = cv2.imread(str(p))
            if t is not None:
                templates.append((p.stem, t, _build_cart_mask(t)))
        _cart_templates = templates
    return _cart_templates


def _cart_search_region(frame, plank_y: int) -> Tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) for the cart-search ROI. The cart can be
    anywhere from the very top of the minigame overlay (peak of a jump)
    down to seated on the plank, so we let the search run from frame top
    to plank_y + 15. The minigame UI is in the upper portion of the
    window so we also cap at plank_y + 15 below to skip the score row
    and lower UI."""
    h, w = frame.shape[:2]
    return (0, 0, w, min(h, plank_y + 15))


def _match_cart(frame, plank_y: int, *, x_center: Optional[int] = None,
                scales: tuple = CART_SCALES) -> Optional[dict]:
    """Multi-template match the cart. Returns {center, half_width, scale,
    template, score} of the best match >= CART_MATCH_THRESHOLD, else None.

    x_center restricts the search to a CART_TRACK_X_MARGIN-wide column
    around it (the cart's x is fixed per run); omit for a full-width
    search. match_multiscale safely skips any scale whose template exceeds
    the (narrow) region, so the column never has to fit the largest scale."""
    templates = _load_cart_templates()
    if not templates:
        return None
    h, w = frame.shape[:2]
    if x_center is not None:
        x0 = max(0, x_center - CART_TRACK_X_MARGIN)
        x1 = min(w, x_center + CART_TRACK_X_MARGIN)
        region = (x0, 0, x1, min(h, plank_y + 15))
    else:
        region = _cart_search_region(frame, plank_y)
    best = None
    for name, t, mask in templates:
        top_left, val, scale = match_multiscale_masked(
            frame, t, mask, region=region, scales=scales)
        if top_left is None:
            continue
        th = int(round(t.shape[0] * scale))
        tw = int(round(t.shape[1] * scale))
        center = (top_left[0] + tw // 2, top_left[1] + th // 2)
        if best is None or val > best["score"]:
            best = {
                "center": center,
                "score": float(val),
                "scale": scale,
                "template": name,
                "half_width": max(15, int(round(t.shape[1] * scale)) // 2),
            }
    if best is None or best["score"] < CART_MATCH_THRESHOLD:
        return None
    return best


def _estimate_cart_half_width(frame, plank_y: int) -> int:
    """Cart half-width via a full match. Retained for find_next_terrain's
    fallback (when callers don't pass cart_right) and any external use;
    the hot path gets half_width from find_cart_detailed instead."""
    res = _match_cart(frame, plank_y)
    return res["half_width"] if res is not None else 30


def _find_plank_top_y(frame) -> Optional[int]:
    """Locate the brightest tan-hued horizontal band — the plank top.

    Restricted to the upper-middle of the window since the minigame
    overlay floats above the player (and the player typically stands in
    the lower half of the world view). PLANK_MIN_SCORE guards against
    overworld tan-textured surfaces being mistaken for the plank when
    the minigame isn't active."""
    h, w = frame.shape[:2]
    x0 = int(PLANK_X0_FRAC * w)
    x1 = int(PLANK_X1_FRAC * w)
    y_min = int(PLANK_Y_FRAC_MIN * h)
    y_max = int(PLANK_Y_FRAC_MAX * h)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    band = hsv[y_min:y_max, x0:x1]
    is_plank = ((band[:, :, 0] >= PLANK_H_LO) & (band[:, :, 0] <= PLANK_H_HI) &
                (band[:, :, 1] >= PLANK_S_MIN) & (band[:, :, 2] >= PLANK_V_MIN))
    score_per_row = is_plank.sum(axis=1)
    best_y = int(np.argmax(score_per_row))
    if score_per_row[best_y] < PLANK_MIN_SCORE:
        return None
    return best_y + y_min


def _scan_plank_pits(frame, plank_y: int,
                     x_start: Optional[int] = None,
                     x_end: Optional[int] = None) -> list[Tuple[int, int]]:
    """Return list of (x_left, x_right) for pits on the plank.

    x_start / x_end constrain the scan to a sub-range. Callers should
    typically pass the cart's right edge as x_start (so the cart itself
    isn't read as a pit) and the result of _find_plank_x_range as x_end
    (so cave wall past the plank isn't either)."""
    h, w = frame.shape[:2]
    x0 = x_start if x_start is not None else int(PLANK_X0_FRAC * w)
    x1 = x_end if x_end is not None else int(PLANK_X1_FRAC * w)
    y0 = plank_y + PIT_SCAN_DY[0]
    y1 = plank_y + PIT_SCAN_DY[1]
    if y0 < 0 or y1 > h or x1 <= x0:
        return []
    band = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    is_pit = hsv[:, :, 2] < PIT_V_MAX
    col_is_pit = is_pit.sum(axis=0) > (y1 - y0) // 2
    return _extract_runs(col_is_pit, offset=x0, min_width=PIT_MIN_WIDTH)


def _scan_plank_ore(frame, plank_y: int,
                    x_start: Optional[int] = None,
                    x_end: Optional[int] = None) -> list[Tuple[int, int]]:
    """Return list of (x_left, x_right) for slam-to-score ore above the plank.

    Disabled — returns [] unconditionally. Neither candidate above-plank
    signal is the real ore (see the ORE_SCAN_DY comment block and
    docs/mining_plan.md Run 6): tan-hue detects the cave wall, blue-hue
    detects static parallax-background crystals. Both are phantom, and a
    phantom obstacle here is worse than none: find_next_terrain returns the
    nearest pit-OR-ore, and the jump trigger fires only on kind=='pit'
    (main.py), so a phantom ore sitting between the cart and a real pit masks
    that pit and suppresses the jump. Against botrun_20260615_012340 the
    blue scan reported "ore at distance 28" every frame from a static
    background pile, which would have hidden every pit and guaranteed a
    death. Until a scoring --watch run reveals the real ore (and a
    scroll-velocity foreground/background filter separates it from
    decoration), detecting nothing is the safe, survival-correct behavior.

    Signature and x_start/x_end convention preserved for the eventual real
    implementation and for find_next_terrain's caller."""
    return []


def _find_plank_x_range(frame, plank_y: int) -> Optional[Tuple[int, int]]:
    """Return (x_left, x_right) of the visible plank — the floating
    minigame overlay is a fixed-pixel-width panel and doesn't extend to
    the window edges. Detected by scanning a thin y-band at the plank
    top for plank-tan-signature columns, clustering them with
    PLANK_CLUSTER_GAP tolerance to bridge real pits, and returning the
    bounds of the largest cluster. Stray tan-bright pixels elsewhere on
    screen (UI, cave decoration) don't extend the range. Returns None if
    no plank found."""
    h, _w = frame.shape[:2]
    y0 = plank_y
    y1 = min(h, plank_y + 3)
    if y1 <= y0:
        return None
    hsv = cv2.cvtColor(frame[y0:y1, :], cv2.COLOR_BGR2HSV)
    is_plank = ((hsv[:, :, 0] >= PLANK_H_LO) & (hsv[:, :, 0] <= PLANK_H_HI) &
                (hsv[:, :, 1] >= PLANK_S_MIN) & (hsv[:, :, 2] >= PLANK_V_MIN))
    col_score = is_plank.sum(axis=0)
    plank_cols = np.where(col_score >= 2)[0]
    if len(plank_cols) < 10:
        return None
    # Cluster into groups separated by gaps > PLANK_CLUSTER_GAP. Real
    # pits are typically 30-80px wide; cave-wall stray pixels live in
    # entirely separate parts of the frame (hundreds of px away).
    clusters: list[list[int]] = [[int(plank_cols[0])]]
    for c in plank_cols[1:]:
        if c - clusters[-1][-1] > PLANK_CLUSTER_GAP:
            clusters.append([int(c)])
        else:
            clusters[-1].append(int(c))
    # Pick the cluster with the most plank columns (densest), not just
    # the widest one — UI decorations can span many columns sparsely.
    best = max(clusters, key=lambda cl: len(cl))
    return best[0], best[-1] + 1


def _extract_runs(col_mask, offset: int, min_width: int,
                  merge_gap: int = MERGE_GAP_PX) -> list[Tuple[int, int]]:
    """Find contiguous True runs in col_mask, return (x_left, x_right)
    pairs offset into frame coords. Adjacent runs separated by <= merge_gap
    columns of False are merged — a single pit's interior often has small
    bright spots (spike highlights) that split it into fragments. Final
    pass filters out runs narrower than min_width."""
    runs: list[Tuple[int, int]] = []
    in_run = False
    start = 0
    for x, p in enumerate(col_mask):
        if p and not in_run:
            in_run, start = True, x
        elif not p and in_run:
            runs.append((start + offset, x + offset))
            in_run = False
    if in_run:
        runs.append((start + offset, len(col_mask) + offset))

    if merge_gap > 0 and len(runs) > 1:
        merged = [runs[0]]
        for a, b in runs[1:]:
            prev_a, prev_b = merged[-1]
            if a - prev_b <= merge_gap:
                merged[-1] = (prev_a, b)
            else:
                merged.append((a, b))
        runs = merged

    return [(a, b) for (a, b) in runs if (b - a) >= min_width]
