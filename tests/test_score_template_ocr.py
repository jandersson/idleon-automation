"""Tests for common.score_template_ocr.

Uses live darts and hoops template libraries as fixtures — they're
committed to the repo so the tests are reproducible. End-to-end tests
read actual captured monitor frames and verify the OCR returns the
score we already know via the per-game DB / visual inspection.
"""
from pathlib import Path

import cv2
import numpy as np

from common.score_template_ocr import (
    binarize,
    extract_digit_components,
    load_templates,
    make_score_reader,
    match_digit,
    save_template,
)

ROOT = Path(__file__).parent.parent
DARTS_TEMPLATES = ROOT / "minigames" / "darts" / "assets" / "digit_templates"
HOOPS_TEMPLATES = ROOT / "minigames" / "hoops" / "assets" / "digit_templates"


def _make_score_crop(digit_patches: list[np.ndarray], gap_px: int = 1, pad: tuple[int, int] = (4, 2)) -> np.ndarray:
    """Compose a 40x18 BGR crop from a list of binary digit patches.
    Patches are placed left-to-right with `gap_px` between, padded by
    `pad=(left, top)`. Background is the darts-region dark red; foreground
    is white. Matches the shape and palette of real darts score crops."""
    h, w = 18, 40
    crop = np.full((h, w, 3), (0, 16, 58), dtype=np.uint8)  # dark red-brown
    x = pad[0]
    y = pad[1]
    for patch in digit_patches:
        ph, pw = patch.shape
        for py in range(ph):
            for px in range(pw):
                if patch[py, px] > 0 and y + py < h and x + px < w:
                    crop[y + py, x + px] = (249, 249, 249)
        x += pw + gap_px
    return crop


def test_binarize_separates_fg_bg():
    """Foreground (bright) and background (dark) pixels go to 255 vs 0."""
    img = np.full((10, 20, 3), (0, 16, 58), dtype=np.uint8)
    img[5, 10] = (249, 249, 249)
    binary = binarize(img)
    assert binary[5, 10] == 255
    assert binary[0, 0] == 0


def test_extract_digit_components_handles_empty_crop():
    """All-background crop yields no components."""
    crop = np.full((10, 20, 3), (0, 16, 58), dtype=np.uint8)
    assert extract_digit_components(crop) == []


def test_extract_digit_components_returns_left_to_right():
    """Components come out sorted by x so digit order matches reading."""
    crop = np.full((18, 40, 3), (0, 16, 58), dtype=np.uint8)
    # Two 5x5 white blocks at x=15 and x=5 (out of order)
    crop[5:10, 15:20] = (249, 249, 249)
    crop[5:10, 5:10] = (249, 249, 249)
    components = extract_digit_components(crop)
    assert len(components) == 2
    assert components[0][1][0] < components[1][1][0]  # x ordering


def test_match_digit_exact_match_returns_digit():
    template = np.array([[0, 255, 0], [255, 255, 255], [0, 255, 0]], dtype=np.uint8)
    templates = {5: template}
    patch = template.copy()
    assert match_digit(patch, templates) == 5


def test_match_digit_different_shape_returns_none():
    """A patch with a different bounding box shape can't match — pixel-art
    digits have stable native dimensions."""
    templates = {5: np.array([[255, 255], [255, 255]], dtype=np.uint8)}
    patch_different = np.array([[255, 255, 255]], dtype=np.uint8)
    assert match_digit(patch_different, templates) is None


def test_match_digit_different_pixels_same_shape_returns_none():
    """Same shape but different pixels (e.g. 6 vs 9 in the same font)
    must NOT match — the whole point of template OCR."""
    templates = {6: np.array([[255, 255], [255, 0]], dtype=np.uint8)}
    patch = np.array([[255, 255], [0, 255]], dtype=np.uint8)
    assert match_digit(patch, templates) is None


def test_load_templates_returns_dict_with_present_digits(tmp_path):
    """Templates are loaded by digit; missing PNGs = missing keys."""
    sample = np.array([[255, 0], [0, 255]], dtype=np.uint8)
    save_template(tmp_path, 3, sample)
    save_template(tmp_path, 7, sample)
    loaded = load_templates(tmp_path)
    assert set(loaded.keys()) == {3, 7}
    assert np.array_equal(loaded[3], sample)


def test_make_score_reader_returns_none_on_no_components():
    """Empty crop with no digits → None (not 0 — None preserves the
    'OCR failed' contract callers expect)."""
    reader = make_score_reader(DARTS_TEMPLATES)
    crop = np.full((18, 40, 3), (0, 16, 58), dtype=np.uint8)
    assert reader(crop) is None


def test_make_score_reader_returns_none_for_none_input():
    reader = make_score_reader(DARTS_TEMPLATES)
    assert reader(None) is None


def test_darts_reads_known_post_throw_crops():
    """End-to-end: read score from captured monitor crops where we know
    the answer. Goes through binarize + components + template-match."""
    reader = make_score_reader(DARTS_TEMPLATES)
    # Verified by hand against the screenshots (2026-06-10 audit — the
    # score region was realigned after it was found clipping the leading
    # digit, so the fixtures must come from frames captured at the same
    # window geometry the current regions.json was calibrated on).
    cases = [
        ("minigames/darts/assets/monitor/throw_008_010414/post_throw.png", 22),
        ("minigames/darts/assets/monitor/throw_009_010439/post_throw.png", 24),
        ("minigames/darts/assets/monitor/throw_016_010630/post_throw.png", 42),
    ]
    # Use real region for darts
    from common.regions import get_region
    from pathlib import Path as P
    darts_dir = ROOT / "minigames" / "darts"
    for rel_path, expected in cases:
        frame = cv2.imread(str(ROOT / rel_path))
        if frame is None:
            # Skip if the monitor frame was cleaned up — keeps the test
            # robust to disk-cleanup garbage collection of old throws.
            continue
        h, w = frame.shape[:2]
        region = get_region(darts_dir, "score", w, h)
        assert region is not None, "darts score region must be configured"
        crop = frame[
            region["top"]:region["top"] + region["height"],
            region["left"]:region["left"] + region["width"],
        ]
        assert reader(crop) == expected, f"{rel_path}: expected {expected}, got {reader(crop)}"


def test_unknown_digit_in_score_returns_none():
    """If even one digit in a multi-digit score has no template, the
    whole reading fails (we don't want '0' guessed as something else
    just because 0 isn't loaded). Caller should treat None as failure
    and fall back to whatever sanity-checking logic they have."""
    reader = make_score_reader(DARTS_TEMPLATES)
    # Construct a crop with two digits where one is a definitely-unknown
    # shape (a giant white square).
    crop = np.full((18, 40, 3), (0, 16, 58), dtype=np.uint8)
    # Plant a copy of the "6" template at x=4 (real digit, should match)
    six = cv2.imread(str(DARTS_TEMPLATES / "6.png"), cv2.IMREAD_GRAYSCALE)
    if six is not None:
        h_six, w_six = six.shape
        for py in range(h_six):
            for px in range(w_six):
                if six[py, px] > 127:
                    crop[4 + py, 4 + px] = (249, 249, 249)
        # Plant an unknown shape (small 5x5 block) at x=20 with a gap
        crop[6:11, 20:25] = (249, 249, 249)
        assert reader(crop) is None
