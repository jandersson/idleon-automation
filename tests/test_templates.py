"""Tests for common/templates.py — multi-scale template matching."""
import cv2
import numpy as np
import pytest

from common.templates import (
    ScaleLockMatcher,
    masked_match_confidence,
    masked_zncc_map,
    match_multiscale,
    match_multiscale_center,
    match_multiscale_masked,
    match_multiscale_zncc_center,
)


def _make_marker(size: tuple[int, int]) -> np.ndarray:
    """A high-contrast pseudo-random marker (deterministic seed) with strong
    internal variance — TM_CCOEFF_NORMED needs non-uniform pixels to score
    matches reliably. Plain white squares fail with zero-variance edge cases.
    """
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)


def _make_image_with_marker(image_size: tuple[int, int], marker: np.ndarray, at: tuple[int, int]):
    img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    x, y = at
    mh, mw = marker.shape[:2]
    img[y:y + mh, x:x + mw] = marker
    return img


def test_finds_marker_at_native_scale():
    marker = _make_marker((40, 40))
    image = _make_image_with_marker((400, 300), marker, at=(100, 80))

    top_left, val, scale = match_multiscale(image, marker, scales=(0.9, 1.0, 1.1))
    assert top_left is not None
    assert abs(top_left[0] - 100) <= 1
    assert abs(top_left[1] - 80) <= 1
    assert val > 0.9
    assert scale == 1.0


def test_finds_marker_at_half_size_template():
    """If the live image's marker is smaller than the captured template,
    multi-scale should find it at scale < 1."""
    template = _make_marker((40, 40))
    # Resize the marker to half size before placing in image.
    half = cv2.resize(template, (20, 20), interpolation=cv2.INTER_AREA)
    image = _make_image_with_marker((400, 300), half, at=(150, 100))

    top_left, val, scale = match_multiscale(image, template, scales=(0.4, 0.5, 0.6, 0.75, 1.0))
    assert top_left is not None
    assert scale == 0.5
    assert abs(top_left[0] - 150) <= 2
    assert abs(top_left[1] - 100) <= 2


def test_returns_none_when_template_bigger_than_image():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    template = _make_marker((40, 40))

    top_left, val, scale = match_multiscale(image, template, scales=(1.0, 1.5))
    assert top_left is None


def test_region_restricts_search():
    """Two markers — restricting the region should pick the in-region one."""
    marker = _make_marker((40, 40))
    image = _make_image_with_marker((400, 300), marker, at=(50, 50))
    image[200:240, 300:340] = marker  # second copy

    # Region covering only the second marker.
    top_left, val, scale = match_multiscale(image, marker, region=(280, 180, 400, 260), scales=(1.0,))
    assert top_left is not None
    assert top_left[0] == 300
    assert top_left[1] == 200


def test_match_multiscale_center_returns_box_center():
    marker = _make_marker((40, 40))
    image = _make_image_with_marker((400, 300), marker, at=(100, 80))

    center, val, scale = match_multiscale_center(image, marker, scales=(1.0,))
    assert center is not None
    # Center of a 40x40 box at (100,80) is (120, 100).
    assert abs(center[0] - 120) <= 1
    assert abs(center[1] - 100) <= 1


def test_scale_lock_locks_after_confident_match():
    marker = _make_marker((40, 40))
    image = _make_image_with_marker((400, 300), marker, at=(100, 80))
    matcher = ScaleLockMatcher(scales=(0.5, 0.75, 1.0, 1.25, 1.5))

    center, val, scale = matcher.match_center(image, marker)
    assert val > 0.9
    assert matcher.locked_scale == 1.0


def test_scale_lock_locked_polls_match_full_sweep():
    """On locked polls the returned (center, conf) must be identical to the
    full sweep whenever the winning scale is inside the locked neighborhood."""
    marker = _make_marker((40, 40))
    image = _make_image_with_marker((400, 300), marker, at=(100, 80))
    scales = (0.5, 0.75, 1.0, 1.25, 1.5)
    matcher = ScaleLockMatcher(scales=scales)

    matcher.match_center(image, marker)  # establishes the lock
    locked = matcher.match_center(image, marker)
    full = match_multiscale_center(image, marker, scales=scales)
    assert locked == full


def test_scale_lock_resync_recovers_from_resize():
    """After the marker changes size (window resize), locked polls miss it,
    but the periodic full sweep re-locks within resync_every polls."""
    marker = _make_marker((40, 40))
    big = _make_image_with_marker((400, 300), marker, at=(100, 80))
    small_marker = cv2.resize(marker, (20, 20), interpolation=cv2.INTER_AREA)
    small = _make_image_with_marker((400, 300), small_marker, at=(150, 100))
    matcher = ScaleLockMatcher(scales=(0.5, 0.75, 1.0, 1.25, 1.5), resync_every=3)

    matcher.match_center(big, marker)
    assert matcher.locked_scale == 1.0
    for _ in range(matcher.resync_every + 1):
        center, val, scale = matcher.match_center(small, marker)
    assert matcher.locked_scale == 0.5
    assert val > 0.9


def test_scale_lock_always_scales_searched_when_locked():
    """A marker matching only at an always_scale is still found on locked
    polls even though it's outside the locked neighborhood."""
    marker = _make_marker((40, 40))
    big = _make_image_with_marker((400, 300), marker, at=(100, 80))
    small_marker = cv2.resize(marker, (20, 20), interpolation=cv2.INTER_AREA)
    small = _make_image_with_marker((400, 300), small_marker, at=(150, 100))
    scales = (0.5, 0.75, 1.0, 1.25, 1.5)

    matcher = ScaleLockMatcher(scales=scales, always_scales=(0.5,), resync_every=100)
    matcher.match_center(big, marker)
    assert matcher.locked_scale == 1.0
    center, val, scale = matcher.match_center(small, marker)
    assert scale == 0.5
    assert val > 0.9

    # Without the always_scale, the same locked poll misses the small marker.
    matcher2 = ScaleLockMatcher(scales=scales, resync_every=100)
    matcher2.match_center(big, marker)
    _, val2, _ = matcher2.match_center(small, marker)
    assert val2 < 0.9


def test_scale_lock_rejects_always_scales_outside_scales():
    with pytest.raises(ValueError):
        ScaleLockMatcher(scales=(0.5, 1.0), always_scales=(0.6,))


def _sprite_with_border():
    """A foreground marker surrounded by a border ring, plus a mask that
    covers only the foreground. The border stands in for the background
    baked into a real sprite crop (e.g. the plank/cave around the mining
    cart)."""
    rng = np.random.default_rng(seed=7)
    fg = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    template = np.zeros((40, 40, 3), dtype=np.uint8)
    template[8:32, 8:32] = fg
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[8:32, 8:32] = 255  # foreground only
    return template, mask, fg


def test_masked_match_is_background_independent():
    """The whole point of the masked cart finder: the same sprite is found
    at a high score on two DIFFERENT backgrounds, where an unmasked match
    collapses when the background changes (the mining cart mid-jump:
    plank → cave+ore behind it). Masked TM_CCORR_NORMED scores only the
    masked foreground, so the background swap doesn't move the score."""
    template, mask, fg = _sprite_with_border()

    def scene(bg_value):
        img = np.full((300, 400, 3), bg_value, dtype=np.uint8)
        # paste the foreground (not the border) at a known spot
        img[100:124, 200:224] = fg
        return img

    bg_a = scene(30)
    bg_b = scene(200)

    _, val_a, _ = match_multiscale_masked(bg_a, template, mask, scales=(1.0,))
    _, val_b, _ = match_multiscale_masked(bg_b, template, mask, scales=(1.0,))
    assert val_a > 0.95
    assert val_b > 0.95
    # The unmasked match, by contrast, is pulled off by the border pixels
    # landing on a different background — it should not stay this confident
    # on both. (Sanity check that the mask is doing the work.)
    _, unmasked_b, _ = match_multiscale(bg_b, template, scales=(1.0,))
    assert unmasked_b < val_b


def test_masked_match_locates_center():
    template, mask, fg = _sprite_with_border()
    img = np.full((300, 400, 3), 40, dtype=np.uint8)
    img[100:124, 200:224] = fg
    top_left, val, scale = match_multiscale_masked(img, template, mask, scales=(1.0,))
    assert top_left is not None
    # foreground sits 8px inside the 40px template, so the template's
    # top-left is 8px up/left of the pasted foreground.
    assert abs(top_left[0] - (200 - 8)) <= 1
    assert abs(top_left[1] - (100 - 8)) <= 1
    assert val > 0.95


def test_masked_zncc_confidence_is_background_independent():
    """The property find_play_button needs: the masked-ZNCC confidence at the
    sprite is high and ~the same over two DIFFERENT backgrounds (the PLAY GAME
    button scores ~0.99 whether it sits over its home map or the desert). The
    foreground is pasted at a known spot so the confidence is measured at the
    true location, isolating background-invariance from localization."""
    template, mask, fg = _sprite_with_border()

    def conf_on(bg_value):
        img = np.full((300, 400, 3), bg_value, dtype=np.uint8)
        img[100:124, 200:224] = fg  # foreground only, on a fresh background
        return masked_match_confidence(img, template, mask, top_left=(192, 92), scale=1.0)

    conf_dark, conf_bright = conf_on(30), conf_on(200)
    assert conf_dark > 0.95 and conf_bright > 0.95
    assert abs(conf_dark - conf_bright) < 0.02  # background swap barely moves it


def test_masked_zncc_center_finds_sprite():
    """End-to-end: locate + confidence on a clean scene returns a centre on
    the sprite at high confidence."""
    template, mask, _ = _sprite_with_border()
    img = np.full((300, 400, 3), 70, dtype=np.uint8)
    img[100:140, 200:240] = template  # full template (border included)
    center, conf, _ = match_multiscale_zncc_center(img, template, mask, scales=(1.0,))
    assert center is not None
    assert abs(center[0] - 220) <= 3 and abs(center[1] - 120) <= 3
    assert conf > 0.99


def test_masked_zncc_rejects_bright_flat_region():
    """The reason ZNCC is used over TM_CCORR_NORMED for the button: mean-
    subtraction makes a bright flat region score ~0, where raw CCORR scores
    it ~0.9 (which collapsed the prompt-vs-gameplay margin)."""
    template, mask, _ = _sprite_with_border()
    flat = np.full((300, 400, 3), 255, dtype=np.uint8)
    conf = masked_match_confidence(flat, template, mask, top_left=(50, 50), scale=1.0)
    assert abs(conf) < 0.1


def test_masked_match_confidence_exact_match_is_one():
    template, mask, _ = _sprite_with_border()
    img = np.full((100, 100, 3), 12, dtype=np.uint8)
    img[20:60, 20:60] = template
    conf = masked_match_confidence(img, template, mask, top_left=(20, 20), scale=1.0)
    assert conf > 0.99


def test_masked_match_confidence_out_of_bounds_is_zero():
    template, mask, _ = _sprite_with_border()
    img = np.full((100, 100, 3), 50, dtype=np.uint8)
    assert masked_match_confidence(img, template, mask, top_left=(80, 80), scale=1.0) == 0.0


# --- masked_zncc_map: the vectorized localizing counterpart -----------------

def _gray_glyph_with_border():
    """A grayscale glyph (the digit-reader case: bright fill, dark outline) with
    a 1px outer border that stands in for background baked into a tight crop, and
    a mask covering the glyph ink (fill + outline) only."""
    rng = np.random.default_rng(seed=11)
    ink = rng.integers(0, 256, size=(8, 6), dtype=np.uint8)
    tmpl = np.zeros((10, 8), dtype=np.uint8)
    tmpl[1:9, 1:7] = ink
    mask = np.zeros((10, 8), dtype=np.uint8)
    mask[1:9, 1:7] = 255
    return tmpl, mask, ink


def test_masked_zncc_map_equals_pointwise_confidence():
    """masked_zncc_map must equal masked_match_confidence at every position —
    it's the same score computed as a full correlation map (so the digit reader
    can localize without a separate confirm pass)."""
    tmpl, mask, _ = _gray_glyph_with_border()
    rng = np.random.default_rng(seed=3)
    img = rng.integers(0, 256, size=(16, 30), dtype=np.uint8)
    zmap = masked_zncc_map(img, tmpl, mask)
    assert zmap.shape == (16 - 10 + 1, 30 - 8 + 1)
    for y in range(zmap.shape[0]):
        for x in range(zmap.shape[1]):
            ref = masked_match_confidence(img, tmpl, mask, (x, y), 1.0)
            assert abs(float(zmap[y, x]) - ref) < 1e-3


def test_masked_zncc_map_peaks_on_the_glyph_over_any_background():
    """The property the score reader needs: the map peaks (~1.0) exactly at the
    glyph location, and that peak is background-invariant — paste the ink over a
    dark and a bright flat background and the peak stays put and high."""
    tmpl, mask, ink = _gray_glyph_with_border()
    for bg in (20, 230):
        img = np.full((16, 30), bg, dtype=np.uint8)
        img[3:11, 10:16] = ink            # ink only; the border stays background
        zmap = masked_zncc_map(img, tmpl, mask)
        py, px = np.unravel_index(int(np.argmax(zmap)), zmap.shape)
        assert float(zmap[py, px]) > 0.99
        assert abs(px - 9) <= 1 and abs(py - 2) <= 1   # template top-left = ink - (1,1)


def test_masked_zncc_map_none_when_template_too_big():
    tmpl, mask, _ = _gray_glyph_with_border()
    assert masked_zncc_map(np.zeros((5, 5), np.uint8), tmpl, mask) is None
