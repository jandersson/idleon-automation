"""Read pixel-art score digits via per-digit template matching.

Tesseract is unreliable on the 8x9-ish digit glyphs used by Idleon's
minigame score regions (e.g. reads "9" as None, "14" as None, "5" as
195 or -4). This module replaces it with exact-binary template matching:
threshold the score crop, find connected components (= individual
digits), and compare each component to a per-game template library.

Each game keeps its own templates because the fonts differ slightly
(hoops' "1" has a flag at top; darts' doesn't). Templates live in
`<game>/assets/digit_templates/<digit>.png` and are loaded once per
call to `make_score_reader`.

The reader returns None when ANY digit can't be matched, preserving the
None-on-failure contract of the old `common.score_ocr.read_score`.
That lets callers' downstream sanity-check (e.g. darts.main's
`_pick_best_score_after`) treat unreadable rows the same as before.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np


# Binarization threshold for the score region. Backgrounds are very dark
# (val ~27-50), foregrounds are near-white (val ~249). 100 sits comfortably
# between with no rendering variance to worry about — pixel art is clean.
BINARIZE_THRESHOLD = 100

# Connected-component size filter to reject noise. Anti-aliasing pixels or
# stray UI edges can produce 1-4 pixel components; real digit strokes are
# at least 5 pixels in a single column.
MIN_COMPONENT_AREA = 5
MIN_COMPONENT_WIDTH = 2
MIN_COMPONENT_HEIGHT = 4


def binarize(crop: np.ndarray) -> np.ndarray:
    """Threshold the score crop to a binary (0/255) digit mask."""
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    _, binary = cv2.threshold(gray, BINARIZE_THRESHOLD, 255, cv2.THRESH_BINARY)
    return binary


def extract_digit_components(crop: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Threshold and extract one binary patch per detected digit.

    Returns a list of (binary_patch, (x, y, w, h)) tuples sorted
    left-to-right by x. Empty list when nothing readable is detected.
    """
    binary = binarize(crop)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    # Label 0 is the background — skip.
    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        if area < MIN_COMPONENT_AREA or w < MIN_COMPONENT_WIDTH or h < MIN_COMPONENT_HEIGHT:
            continue
        # Use the labels mask (not the raw binary) so a fat component
        # doesn't accidentally include pixels from a neighbouring digit
        # at the same y range.
        patch = ((labels[y:y + h, x:x + w] == label_id).astype(np.uint8)) * 255
        components.append((patch, (int(x), int(y), int(w), int(h))))
    # Digits in a score readout share one glyph height; anything much
    # shorter is punctuation or noise that slipped past the absolute
    # minima — e.g. the bottom dot of the "Score:" colon (6px) next to
    # 9px digits when a region crop starts near the label (darts,
    # 2026-06-10). Filter relative to the tallest component.
    if components:
        max_h = max(c[1][3] for c in components)
        components = [c for c in components if c[1][3] >= 0.75 * max_h]
    components.sort(key=lambda c: c[1][0])
    return components


def match_digit(patch: np.ndarray, templates: dict[int, np.ndarray]) -> int | None:
    """Find the digit whose template matches `patch` pixel-for-pixel.

    Templates are exact binary patterns at native size; matching requires
    identical shape AND identical pixels. Pixel-art digits render
    deterministically so this works. Returns None when no template
    matches.
    """
    for digit, template in templates.items():
        if patch.shape == template.shape and np.array_equal(patch, template):
            return digit
    return None


def load_templates(template_dir: Path) -> dict[int, np.ndarray]:
    """Load digit templates from <template_dir>/<digit>.png. Each PNG is
    expected to be a binarized digit (single-digit pixel patch saved by
    `save_template`). Missing digits = missing dict entries; OCR will
    return None on any unreadable input until they're captured."""
    templates: dict[int, np.ndarray] = {}
    for digit in range(10):
        path = template_dir / f"{digit}.png"
        if not path.exists():
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Re-binarize to guarantee 0/255 even if the PNG round-tripped
        # through a lossy intermediate.
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        templates[digit] = binary
    return templates


def save_template(template_dir: Path, digit: int, patch: np.ndarray) -> Path:
    """Save a digit template PNG. Caller is responsible for choosing the
    canonical patch (e.g. the first observation of that digit)."""
    template_dir.mkdir(parents=True, exist_ok=True)
    path = template_dir / f"{digit}.png"
    cv2.imwrite(str(path), patch)
    return path


def make_score_reader(template_dir: Path) -> Callable[[np.ndarray | None], int | None]:
    """Build a `read_score(crop) -> int | None` bound to the given
    template library. Templates load once at make-time, not per call."""
    templates = load_templates(template_dir)

    def read_score(crop: np.ndarray | None) -> int | None:
        if crop is None:
            return None
        components = extract_digit_components(crop)
        if not components:
            return None
        digit_chars: list[str] = []
        for patch, _box in components:
            d = match_digit(patch, templates)
            if d is None:
                return None
            digit_chars.append(str(d))
        if not digit_chars:
            return None
        try:
            return int("".join(digit_chars))
        except ValueError:
            return None

    return read_score
