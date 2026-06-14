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
