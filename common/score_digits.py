"""Tolerant pixel-digit score reader, shared across minigames.

Idleon's minigame scores ("N PTS") render as white-filled, dark-outlined
pixel digits on a VARYING background (light sky, dark banner, tan dock,
water). Tesseract reads these ~10px stylized glyphs as garbage, and exact
binary-pattern matching misses because the stable feature — the dark glyph
OUTLINE — binarizes with 1-2px variance against the non-uniform background.

A resized, zero-mean normalized-correlation match is stable instead: the
correct digit scores ~1.0, the wrong digit ~0.1. So this reader:

  1. binarizes via Otsu keeping the MINORITY class (the digit — an outline on
     a light bg, a fill on a dark bg — is always the minority of a tight score
     crop), so it adapts to the background instead of a fixed threshold;
  2. isolates digit-height connected components left-to-right;
  3. matches each tolerantly against a per-digit template library (resized
     correlation), walking left-to-right and STOPPING at the first component
     that isn't a digit — that's how the trailing "PTS"/"BEST" label (and
     noise) is excluded without relying on it merging into one blob.

The font is fixed per game but DIFFERS between minigames, so each bot keeps
its own ``digit_templates/`` dir (``<digit>.png`` primary, ``<digit>_<tag>.png``
background variants). An unreadable / not-yet-captured digit yields None for
that sample — safe for max-tracking callers, which must never record a
false-high. ``parse_pts`` is the pure text->int helper.

Lifted from the catching bot's reader (catching/score.py) so fishing can reuse
it; catching re-exports these names for back-compat.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

# Connected-component filters. A digit glyph is ~6-10px wide, ~10px tall; a
# merged "PTS"/"BEST" label is much wider. Reject specks below the minima and
# anything much wider than a single digit.
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
# Accept a digit at/above this zero-mean normalized correlation. True ~1.0,
# wrong ~0.1, so 0.6 separates cleanly while rejecting the label/letters.
DIGIT_MATCH_THRESHOLD = 0.6

# A score banner's underside can render as a near-full-width horizontal line
# just above the digits; if the crop catches it, it connects to the digit tops
# and swallows them. Zero any row whose foreground fraction exceeds this — a
# real digit's row never fills the crop width.
HLINE_ROW_FILL = 0.8


def _to_gray(crop: np.ndarray) -> np.ndarray:
    """Accept a gray, BGR, or BGRA crop; return single-channel gray."""
    if crop.ndim == 2:
        return crop
    if crop.shape[2] == 4:
        return cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def _strip_horizontal_lines(binary: np.ndarray) -> np.ndarray:
    """Zero rows that are near-fully foreground (a banner/platform line), so it
    can't connect to and swallow the digits beneath it."""
    if binary.size == 0:
        return binary
    row_fill = (binary > 0).mean(axis=1)
    binary = binary.copy()
    binary[row_fill > HLINE_ROW_FILL] = 0
    return binary


def _binarize_glyph(gray: np.ndarray) -> np.ndarray:
    """Isolate the digit glyph as foreground, adapting to the background via
    Otsu + minority-class selection (the digit is the minority of a tight score
    crop). Strips a horizontal banner line so it can't swallow the digits."""
    if gray.size == 0:
        return gray
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float((binary > 0).mean()) > 0.5:   # foreground is the majority -> flip
        binary = cv2.bitwise_not(binary)
    return _strip_horizontal_lines(binary)


def digit_components(crop: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Isolate digit-height components from a score-region crop.

    Returns ``[(patch, (x, y, w, h)), ...]`` sorted left-to-right, each patch a
    binary (0/255) glyph. Components are filtered to plausible single-digit
    size; the trailing label is excluded later by failing to match as a digit,
    but absurdly wide blobs are dropped so they don't waste a match.
    """
    binary = _binarize_glyph(_to_gray(crop))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    comps: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for label_id in range(1, num_labels):  # 0 is background
        x, y, w, h, area = stats[label_id]
        if area < DIGIT_MIN_AREA or w < DIGIT_MIN_WIDTH or h < DIGIT_MIN_HEIGHT:
            continue
        if w > DIGIT_MAX_WIDTH:
            continue  # the merged label blob, not a single digit
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
    return cv2.resize(patch.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)


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

    ``canon_templates`` maps each digit to a LIST of canonical variants — the
    same digit renders differently across backgrounds, so each is kept and the
    patch matches if ANY correlates. Returns ``(digit, score)`` for the best
    variant, or ``(None, score)`` below ``DIGIT_MATCH_THRESHOLD``.
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

    Walks the digit-height components left-to-right and matches each; the score
    is the leading run of matched digits, stopping at the first component that
    isn't a digit (the label, noise, or an uncaptured glyph).
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
    """Leading integer of an "N PTS" readout ("1 PTS"->1, " 7 BEST "->7), or
    None. Pure; reusable by any text-producing path (e.g. a tesseract fallback)."""
    if not text:
        return None
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


def load_digit_templates(template_dir: Path) -> dict[int, list[np.ndarray]]:
    """Load native binary digit patches, keyed digit -> list of variants. A
    ``<digit>.png`` is the primary variant; ``<digit>_<tag>.png`` adds a
    background-specific one. Missing digits are absent (reader returns None on
    inputs needing them)."""
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
    """Save a native binary digit patch to ``<dir>/<digit>.png`` (or
    ``<dir>/<digit>_<tag>.png`` for a background variant)."""
    template_dir.mkdir(parents=True, exist_ok=True)
    name = f"{digit}.png" if tag is None else f"{digit}_{tag}.png"
    path = template_dir / name
    cv2.imwrite(str(path), patch)
    return path


def make_pts_reader(template_dir: Path) -> Callable[[np.ndarray | None], int | None]:
    """Build a ``read_pts(crop) -> int | None`` bound to a template library.
    Templates load and canonicalize once at make-time, not per call."""
    canon = {d: [_canon(t) for t in variants]
             for d, variants in load_digit_templates(template_dir).items()}

    def read_pts(crop: np.ndarray | None) -> int | None:
        if crop is None or crop.size == 0:
            return None
        return read_pts_from_crop(crop, canon)

    return read_pts
