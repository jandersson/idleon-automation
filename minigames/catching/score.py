"""Read the PTS score from the catching HUD via tolerant digit templates.

Catching's score ("N PTS") renders as white-filled, dark-outlined pixel
digits on the LIGHT sky, unlike hoops'/darts' bright digits on a dark
background. That difference makes both of the existing score readers
unsuitable here, as verified on the 2026-06-17 observe frames:

- ``common.score_ocr`` (tesseract) reads these ~10px stylized glyphs as
  nothing / garbage -- the same failure that motivated the template reader
  for the other bots.
- ``common.score_template_ocr`` matches EXACT binary patterns on a
  bright-foreground binarization. Catching's stable feature is instead the
  dark glyph OUTLINE, and that outline binarizes with 1-2px variance against
  the non-uniform sky, so exact-pixel matching misses (a "0" ring was
  self-inconsistent at IoU 0.40 across frames). A resized, zero-mean
  normalized-correlation match is stable: the correct digit scores ~1.00,
  the wrong digit ~0.10.

So this reader:
  1. inverted-binarizes (dark outline -> foreground),
  2. isolates digit-height components left-to-right, and
  3. matches each tolerantly against a per-digit template library, walking
     left-to-right and STOPPING at the first component that doesn't match a
     digit. That stop is how the trailing "PTS" label (and any noise) is
     excluded -- it does not rely on PTS merging into a single blob.

The library is bootstrapped with 0 and 1 from the observe frames; the rest
are captured live with ``catching-capture-digits`` (the font is fixed, so
each digit needs capturing once). An unreadable or not-yet-captured digit
yields None for that sample -- safe for the max-PTS tracking in main.py,
which must never record a false-high.

``parse_pts`` is the pure text->int helper (leading integer of an "N PTS"
string); unit-tested in tests/test_catching_score.py.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

# The digit is binarized with Otsu, keeping the MINORITY class as foreground.
# This adapts to the background instead of a fixed threshold: on the light sky
# the digit's dark OUTLINE is the minority (so it's the glyph); on the dark,
# desert-tinted banner the bright FILL is the minority (so that's the glyph).
# Either way the glyph is isolated cleanly — the two backgrounds rendered the
# SAME font as opposite-polarity shapes, which a fixed threshold can't span.
# (The differing shapes are handled by per-digit template variants, below.)

# Connected-component filters. A digit glyph is ~6-10px wide, ~10px tall;
# the merged "PTS" label is ~17px wide. Reject specks below the minima and
# treat anything much wider than a single digit as not-a-digit.
DIGIT_MIN_AREA = 5
DIGIT_MIN_WIDTH = 2
DIGIT_MIN_HEIGHT = 4
DIGIT_MAX_WIDTH = 13
# Digits in a readout share one glyph height; keep only components at least
# this fraction of the tallest, dropping the "0" ring's inner speck etc.
DIGIT_HEIGHT_FRAC = 0.6

# Canonical (h, w) every component and template is resized to before the
# correlation compare, so a 1-2px binarization wobble doesn't matter.
DIGIT_CANON_HW = (16, 12)
# Accept a digit match at/above this zero-mean normalized correlation. The
# true digit scores ~1.0 and the wrong digit ~0.1 on the observe frames, so
# 0.6 separates cleanly while still rejecting the "PTS" blob and letters.
DIGIT_MATCH_THRESHOLD = 0.6

# The score banner's tan underside renders as a near-full-width horizontal
# line just above the digits. If the crop catches it, it CONNECTS to the
# digit tops and swallows them into one giant component. Zero any row whose
# foreground fraction exceeds this -- a real digit's row never fills the
# crop width (a "1" is a thin bar; a "0" is two edges), so this strips the
# platform line without touching glyphs, freeing the crop's top margin.
HLINE_ROW_FILL = 0.8


def _to_gray(crop: np.ndarray) -> np.ndarray:
    """Accept a gray, BGR, or BGRA crop; return single-channel gray."""
    if crop.ndim == 2:
        return crop
    if crop.shape[2] == 4:
        return cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def _strip_horizontal_lines(binary: np.ndarray) -> np.ndarray:
    """Zero rows that are near-fully foreground (the banner's platform line),
    so it can't connect to and swallow the digits beneath it."""
    if binary.size == 0:
        return binary
    row_fill = (binary > 0).mean(axis=1)
    binary = binary.copy()
    binary[row_fill > HLINE_ROW_FILL] = 0
    return binary


def _binarize_glyph(gray: np.ndarray) -> np.ndarray:
    """Isolate the digit glyph as foreground, adapting to the background via
    Otsu + minority-class selection (the digit — outline on a light bg, fill
    on a dark bg — is always the minority of a tight score crop). Strips the
    banner's platform line so it can't swallow the digits."""
    if gray.size == 0:
        return gray
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float((binary > 0).mean()) > 0.5:   # foreground is the majority -> flip
        binary = cv2.bitwise_not(binary)
    return _strip_horizontal_lines(binary)


def digit_components(crop: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Isolate digit-height components from a score-region crop.

    Returns ``[(patch, (x, y, w, h)), ...]`` sorted left-to-right, where
    each patch is the binary (0/255) glyph for one component. Components are
    filtered to plausible single-digit size; the trailing "PTS" label is
    NOT removed here (the reader excludes it by failing to match it as a
    digit -- see module docstring), but absurdly wide blobs are dropped so
    they don't waste a match.
    """
    binary = _binarize_glyph(_to_gray(crop))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comps: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for label_id in range(1, num_labels):  # 0 is background
        x, y, w, h, area = stats[label_id]
        if area < DIGIT_MIN_AREA or w < DIGIT_MIN_WIDTH or h < DIGIT_MIN_HEIGHT:
            continue
        if w > DIGIT_MAX_WIDTH:
            continue  # the merged "PTS" blob, not a single digit
        patch = ((labels[y:y + h, x:x + w] == label_id).astype(np.uint8)) * 255
        comps.append((patch, (int(x), int(y), int(w), int(h))))
    if not comps:
        return []
    max_h = max(c[1][3] for c in comps)
    comps = [c for c in comps if c[1][3] >= DIGIT_HEIGHT_FRAC * max_h]
    comps.sort(key=lambda c: c[1][0])
    return comps


def _canon(patch: np.ndarray) -> np.ndarray:
    """Resize a binary patch to the canonical size as float32 for matching."""
    h, w = DIGIT_CANON_HW
    resized = cv2.resize(patch.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    return resized


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean normalized correlation of two equal-shape float arrays."""
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom == 0.0:
        return 0.0
    return float((a * b).sum() / denom)


def match_digit(patch: np.ndarray, canon_templates: dict[int, list[np.ndarray]]) -> tuple[int | None, float]:
    """Best-matching digit for a component patch, by canonical correlation.

    ``canon_templates`` maps each digit to a LIST of canonical template
    variants — the same digit renders as different shapes across backgrounds
    (a dark outline on the sky, a bright fill on the banner), so each is kept
    as a variant and the patch matches if ANY variant correlates. Returns
    ``(digit, score)`` for the best variant, or ``(None, score)`` when below
    ``DIGIT_MATCH_THRESHOLD`` (no template, an uncaptured digit, or "PTS").
    """
    if not canon_templates:
        return None, 0.0
    c = _canon(patch)
    best_digit, best_score = None, -1.0
    for digit, variants in canon_templates.items():
        for tmpl in variants:
            s = _corr(c, tmpl)
            if s > best_score:
                best_digit, best_score = digit, s
    if best_score < DIGIT_MATCH_THRESHOLD:
        return None, best_score
    return best_digit, best_score


def read_pts_from_crop(crop: np.ndarray, canon_templates: dict[int, list[np.ndarray]]) -> int | None:
    """Read the leading number from a score-region crop, or None.

    Walks the digit-height components left-to-right and matches each; the
    score is the leading run of matched digits, stopping at the first
    component that isn't a digit (the "PTS" label, noise, or an uncaptured
    glyph). Returns None when no leading digit matches.
    """
    digits: list[str] = []
    for patch, _box in digit_components(crop):
        digit, _score = match_digit(patch, canon_templates)
        if digit is None:
            break
        digits.append(str(digit))
    if not digits:
        return None
    return int("".join(digits))


def parse_pts(text: str | None) -> int | None:
    """Extract the leading integer from an OCR'd "N PTS" readout.

    Pure helper: the number is always the leftmost token, so the first digit
    run is the score ("1 PTS" -> 1, "10PTS" -> 10, " 7 BEST " -> 7). Returns
    None when there's no digit. Kept independent of the template reader so
    any text-producing path (a future tesseract fallback) can reuse it, and
    so the parse is unit-testable on its own.
    """
    if not text:
        return None
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


def load_digit_templates(template_dir: Path) -> dict[int, list[np.ndarray]]:
    """Load native binary digit patches, keyed by digit -> list of variants.

    A file ``<digit>.png`` is the primary variant; ``<digit>_<tag>.png`` adds
    a background-specific variant (e.g. ``1_dark.png`` for the banner fill vs
    ``1.png`` for the sky outline). Missing digits are absent from the dict --
    the reader returns None on any input needing them until they're captured.
    """
    templates: dict[int, list[np.ndarray]] = {}
    for path in sorted(template_dir.glob("*.png")):
        digit_str = path.stem.split("_")[0]
        if not digit_str.isdigit():
            continue
        digit = int(digit_str)
        if not 0 <= digit <= 9:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        templates.setdefault(digit, []).append(binary)
    return templates


def save_digit_template(template_dir: Path, digit: int, patch: np.ndarray,
                        tag: str | None = None) -> Path:
    """Save a native binary digit patch to ``<dir>/<digit>.png``, or
    ``<dir>/<digit>_<tag>.png`` for a background-specific variant."""
    template_dir.mkdir(parents=True, exist_ok=True)
    name = f"{digit}.png" if tag is None else f"{digit}_{tag}.png"
    path = template_dir / name
    cv2.imwrite(str(path), patch)
    return path


def make_pts_reader(template_dir: Path) -> Callable[[np.ndarray | None], int | None]:
    """Build a ``read_pts(crop) -> int | None`` bound to a template library.

    Templates load and canonicalize once at make-time, not per call.
    """
    canon = {d: [_canon(t) for t in variants]
             for d, variants in load_digit_templates(template_dir).items()}

    def read_pts(crop: np.ndarray | None) -> int | None:
        if crop is None or crop.size == 0:
            return None
        return read_pts_from_crop(crop, canon)

    return read_pts
