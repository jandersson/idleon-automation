import cv2
import numpy as np
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"


def _load(name: str) -> np.ndarray:
    path = ASSETS / name
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Template not found: {path}")
    return img


def _match_in_region(
    frame_bgr: np.ndarray,
    template: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    threshold: float,
) -> tuple[tuple[int, int] | None, float]:
    """Template-match `template` within frame_bgr[y0:y1, x0:x1].

    Returns (center_xy_in_full_frame, confidence) or (None, confidence).
    """
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.shape[0] < template.shape[0] or crop.shape[1] < template.shape[1]:
        return None, 0.0

    result = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < threshold:
        return None, max_val

    th, tw = template.shape[:2]
    cx = x0 + max_loc[0] + tw // 2
    cy = y0 + max_loc[1] + th // 2
    return (cx, cy), max_val


def find_hoop(
    frame: np.ndarray, threshold: float = 0.7
) -> tuple[tuple[int, int] | None, float]:
    """Find the basketball hoop — searched in the right half of the frame."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    h, w = bgr.shape[:2]
    template = _load("hoop.png")
    return _match_in_region(bgr, template, w // 2, 0, w, h, threshold)


def find_platform(
    frame: np.ndarray, threshold: float = 0.7
) -> tuple[tuple[int, int] | None, float]:
    """Find the character's platform — searched in the left half of the frame."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    h, w = bgr.shape[:2]
    template = _load("platform.png")
    return _match_in_region(bgr, template, 0, 0, w // 2, h, threshold)
