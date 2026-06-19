"""Multi-scale template matching helpers.

cv2.matchTemplate isn't scale-invariant: a template captured at one window
resolution won't match the same sprite when the window is resized. These
helpers try several scales and return the best match across all of them.
"""
import cv2
import numpy as np

DEFAULT_SCALES = (0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5)


def match_multiscale(
    image: np.ndarray,
    template: np.ndarray,
    region: tuple[int, int, int, int] | None = None,
    scales: tuple[float, ...] = DEFAULT_SCALES,
) -> tuple[tuple[int, int] | None, float, float]:
    """Try matching `template` at multiple scales within `image`.

    Args:
        image: BGR or BGRA → caller should convert to BGR before calling.
        template: BGR.
        region: optional (x0, y0, x1, y1) crop within image to search.
        scales: scale factors to try (1.0 = original template size).

    Returns:
        (top_left, max_val, best_scale). top_left is in the *image* coordinate
        system (not the cropped region). Returns (None, max_val, best_scale)
        if no scale fit (template too big for image at any scale).
    """
    if region is not None:
        x0, y0, x1, y1 = region
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(image.shape[1], x1)
        y1 = min(image.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            return None, 0.0, 1.0
        crop = image[y0:y1, x0:x1]
    else:
        x0, y0 = 0, 0
        crop = image

    best_val = -1.0
    best_loc: tuple[int, int] | None = None
    best_scale = 1.0

    for scale in scales:
        new_w = max(1, int(round(template.shape[1] * scale)))
        new_h = max(1, int(round(template.shape[0] * scale)))
        if new_w > crop.shape[1] or new_h > crop.shape[0]:
            continue
        scaled = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        result = cv2.matchTemplate(crop, scaled, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_val:
            best_val = max_val
            best_loc = (x0 + max_loc[0], y0 + max_loc[1])
            best_scale = scale

    return best_loc, max(best_val, 0.0), best_scale


def match_multiscale_center(
    image: np.ndarray,
    template: np.ndarray,
    region: tuple[int, int, int, int] | None = None,
    scales: tuple[float, ...] = DEFAULT_SCALES,
) -> tuple[tuple[int, int] | None, float, float]:
    """Like match_multiscale but returns the center of the matched region in
    image coords (and the best scale, so callers can compute the matched-region
    size if needed)."""
    top_left, val, scale = match_multiscale(image, template, region, scales)
    if top_left is None:
        return None, val, scale
    th = int(round(template.shape[0] * scale))
    tw = int(round(template.shape[1] * scale))
    return (top_left[0] + tw // 2, top_left[1] + th // 2), val, scale


def match_multiscale_masked(
    image: np.ndarray,
    template: np.ndarray,
    mask: np.ndarray,
    region: tuple[int, int, int, int] | None = None,
    scales: tuple[float, ...] = DEFAULT_SCALES,
) -> tuple[tuple[int, int] | None, float, float]:
    """Like match_multiscale but matches only the masked (non-zero) template
    pixels via TM_CCORR_NORMED, so background pixels under the template don't
    affect the score.

    Use this when the sprite is rigid but its background changes between
    frames (e.g. the mining cart mid-jump: the cave wall and ore scroll
    behind it, which collapses an unmasked TM_CCOEFF_NORMED match below
    threshold even though the cart sprite itself is unchanged). The mask
    must be the template's size; it's resized alongside the template at each
    scale.

    Returns (top_left, max_val, best_scale) in image coords, same contract as
    match_multiscale. Non-finite correlation cells (TM_CCORR_NORMED divides
    by the masked-patch norm, which is 0 over a flat region) are treated as
    no-match so they never win.
    """
    if region is not None:
        x0, y0, x1, y1 = region
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(image.shape[1], x1)
        y1 = min(image.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            return None, 0.0, 1.0
        crop = image[y0:y1, x0:x1]
    else:
        x0, y0 = 0, 0
        crop = image

    best_val = -1.0
    best_loc: tuple[int, int] | None = None
    best_scale = 1.0

    for scale in scales:
        new_w = max(1, int(round(template.shape[1] * scale)))
        new_h = max(1, int(round(template.shape[0] * scale)))
        if new_w > crop.shape[1] or new_h > crop.shape[0]:
            continue
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        scaled = cv2.resize(template, (new_w, new_h), interpolation=interp)
        scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=interp)
        result = cv2.matchTemplate(crop, scaled, cv2.TM_CCORR_NORMED, mask=scaled_mask)
        result[~np.isfinite(result)] = -1.0
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_val:
            best_val = max_val
            best_loc = (x0 + max_loc[0], y0 + max_loc[1])
            best_scale = scale

    return best_loc, max(best_val, 0.0), best_scale


def masked_match_confidence(
    image: np.ndarray,
    template: np.ndarray,
    mask: np.ndarray,
    top_left: tuple[int, int],
    scale: float,
) -> float:
    """Zero-mean normalized cross-correlation (ZNCC) over the masked template
    pixels at a fixed location/scale.

    A background-invariant CONFIDENCE to pair with match_multiscale's
    localization. Two reasons it beats TM_CCORR_NORMED-with-mask for a UI
    sprite over changing scenery: mean-subtraction discriminates the sprite
    from bright flat regions (raw CCORR scores those ~0.9), and the mask drops
    the pixels that vary with the world behind the overlay. So the same sprite
    scores ~1.0 over any background while non-sprite scenery scores near 0.

    Returns ZNCC in [-1, 1]; 0.0 when the patch is out of bounds, the mask is
    near-empty, or the masked patch is flat (mean-subtraction kills it).
    """
    th = int(round(template.shape[0] * scale))
    tw = int(round(template.shape[1] * scale))
    if th < 1 or tw < 1:
        return 0.0
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    t = cv2.resize(template, (tw, th), interpolation=interp)
    m = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
    x, y = top_left
    patch = image[y:y + th, x:x + tw]
    if patch.shape[:2] != (th, tw):
        return 0.0
    sel = m > 0
    if int(sel.sum()) < 10:
        return 0.0
    a = t[sel].astype(np.float32).ravel()
    b = patch[sel].astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom == 0.0:
        return 0.0
    return float((a * b).sum() / denom)


def masked_zncc_map(
    image_gray: np.ndarray,
    template_gray: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray | None:
    """Masked ZNCC at EVERY top-left position, vectorized — the localizing
    counterpart to masked_match_confidence (which scores one fixed location).

    Same background-invariant score (mean-subtracted NCC over the masked
    template pixels, the world behind the overlay dropped), but computed as a
    full correlation map so a caller can find where a glyph/sprite sits without
    a separate localization pass. Single-channel (grayscale) only: the masked
    digit glyphs are colourless (white fill, dark outline), so intensity carries
    all the structure; pass a gray image + gray template + 0/255 mask.

    Returns a float32 map of ZNCC in [-1, 1], shape
    ``(H - th + 1, W - tw + 1)`` (same as cv2.matchTemplate), or None when the
    template is larger than the image or the masked template is flat (no
    variance to correlate). Positions where the masked image patch is flat read
    0.0 (mean-subtraction kills them), exactly like masked_match_confidence.

    Identity used (means are over the N masked pixels, per output position):
      ZNCC = Σ_mask (T-T̄)(I-Ī) / (‖T-T̄‖ · ‖I-Ī‖)
    The numerator is ``correlate(I, Mc)`` with ``Mc = mask·(T-T̄)`` (the masked,
    centred template, zero outside the mask); ‖I-Ī‖ per position comes from the
    masked running sum and sum-of-squares (``correlate(I, M)`` and
    ``correlate(I², M)``). Verified equal to masked_match_confidence to ~1e-4.
    """
    img = image_gray.astype(np.float32)
    tmpl = template_gray.astype(np.float32)
    m = (mask > 0).astype(np.float32)
    n = float(m.sum())
    if (n < 1 or img.shape[0] < tmpl.shape[0] or img.shape[1] < tmpl.shape[1]):
        return None
    tmpl_mean = float((m * tmpl).sum() / n)
    tmpl_c = m * (tmpl - tmpl_mean)
    tmpl_norm = float(np.sqrt((tmpl_c * tmpl_c).sum()))
    if tmpl_norm == 0.0:
        return None
    cross = cv2.matchTemplate(img, tmpl_c, cv2.TM_CCORR)
    sum_i = cv2.matchTemplate(img, m, cv2.TM_CCORR)
    sum_i2 = cv2.matchTemplate(img * img, m, cv2.TM_CCORR)
    var_i = sum_i2 - (sum_i * sum_i) / n
    denom = tmpl_norm * np.sqrt(np.maximum(var_i, 0.0))
    out = np.zeros_like(cross)
    nz = denom > 1e-6
    out[nz] = cross[nz] / denom[nz]
    return out


def match_multiscale_zncc_center(
    image: np.ndarray,
    template: np.ndarray,
    mask: np.ndarray,
    region: tuple[int, int, int, int] | None = None,
    scales: tuple[float, ...] = DEFAULT_SCALES,
) -> tuple[tuple[int, int] | None, float, float]:
    """Locate `template` with the unmasked multiscale CCOEFF sweep, then
    re-score the winning location with background-invariant masked ZNCC
    (masked_match_confidence). Returns (center, zncc_confidence, scale) — use
    the ZNCC as the accept/reject confidence, NOT the localization value.

    The CCOEFF sweep localizes the sprite well even with the background
    contaminating its score; the masked ZNCC then yields a confidence that
    doesn't sag when the same sprite sits over a different world — which is
    what drops a plain unmasked match below threshold across game regions
    (catching's PLAY GAME prompt: 1.00 in its home map, 0.78 in the desert).
    """
    top_left, _val, scale = match_multiscale(image, template, region, scales)
    if top_left is None:
        return None, 0.0, scale
    conf = masked_match_confidence(image, template, mask, top_left, scale)
    th = int(round(template.shape[0] * scale))
    tw = int(round(template.shape[1] * scale))
    return (top_left[0] + tw // 2, top_left[1] + th // 2), conf, scale


class ScaleLockMatcher:
    """Stateful multi-scale matcher for per-poll hot paths.

    The winning template scale tracks the window size — constant within
    a session except across window resizes — so most polls don't need
    the full scale sweep. After a confident match locks the scale,
    polls match only the locked scale, its immediate neighbors, and
    `always_scales`; every `resync_every`-th poll runs the full sweep,
    so a window resize (or a junk-induced wrong lock) is corrected
    within at most that many polls.

    Confidences are exact full-resolution match values over the scales
    tried — never rescaled or approximated. `always_scales` is for
    callers whose downstream logic is calibrated against the no-match
    noise floor (e.g. an adaptive threshold tracking sub-threshold
    confidence between matches): pass the scale(s) that empirically
    produce that floor so its level is preserved on locked polls.
    Darts evidence (2026-06-10, scripts/probe_pose_scales.py): genuine
    poses win at 0.9-1.1 while the 0.55-0.65 junk floor wins at 0.6 in
    ~97% of frames — so darts locks with always_scales=(0.6,).
    """

    def __init__(
        self,
        scales: tuple[float, ...] = DEFAULT_SCALES,
        lock_threshold: float = 0.7,
        always_scales: tuple[float, ...] = (),
        resync_every: int = 20,
    ):
        if not set(always_scales) <= set(scales):
            raise ValueError("always_scales must be a subset of scales")
        self.scales = tuple(scales)
        self.lock_threshold = lock_threshold
        self.always_scales = tuple(always_scales)
        self.resync_every = resync_every
        self.locked_scale: float | None = None
        self._polls_since_sweep = 0

    def match_center(
        self,
        image: np.ndarray,
        template: np.ndarray,
        region: tuple[int, int, int, int] | None = None,
    ) -> tuple[tuple[int, int] | None, float, float]:
        """Same contract as match_multiscale_center."""
        if self.locked_scale is None or self._polls_since_sweep >= self.resync_every:
            scales = self.scales
            self._polls_since_sweep = 0
        else:
            i = self.scales.index(self.locked_scale)
            scales = tuple(sorted({
                *self.scales[max(0, i - 1):i + 2], *self.always_scales,
            }))
            self._polls_since_sweep += 1
        center, val, scale = match_multiscale_center(image, template, region, scales)
        if center is not None and val >= self.lock_threshold:
            self.locked_scale = scale
        return center, val, scale
