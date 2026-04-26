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
