"""Extract dart trajectory metrics from per-throw flight frames.

The darts bot captures flight_*.png at 50ms intervals during the
post-throw cooldown. Re-running a gray-pixel + motion-masked detector
across those frames gives us a (frame_idx, x, y) trajectory, which we
summarise into:

- launch_angle_deg : angle (in degrees, 0 = right, +ve = downward in
                     screen coords) of the dart's velocity vector over
                     the first ~3 detected frames. None if the dart
                     wasn't seen at least twice early in the flight.
- apex_y           : the highest point reached (smallest y)
- landing_x        : x of the last detected position before the dart
                     leaves the visible area
- frames_seen      : how many frames the dart was detected in (signal
                     quality — < 3 means low-confidence trajectory)

These mirror common.ball_trajectory's shape so the same downstream
patterns (per-shot logging, predictor training) carry over.

Detection strategy: gray pixels (low saturation, mid-to-high value) +
motion mask vs. previous frame, then connected-components filtered to
the small-and-horizontal-ish blobs. The held dart in the player's
hand is mostly stationary so the motion mask filters it out; static
gray pegs on the right wall are filtered the same way.
"""
from pathlib import Path

import cv2
import numpy as np


# Gray-pixel HSV bounds — same heuristic that auto-extracted the
# release template (low sat, decent value). Wider than necessary so
# we tolerate background variation across teleport positions.
_GRAY_SAT_MAX = 60
_GRAY_VAL_MIN = 130

# Min/max pixel area for a dart blob. Small lower bound because the
# dart is only ~40x9 px and motion masking shaves both edges, leaving
# a thin sliver. Upper bound rejects the held-dart-plus-arm or any
# cosmetics blob that happens to be gray.
_DART_AREA_MIN = 6
_DART_AREA_MAX = 200

# Inter-frame absdiff threshold for "this pixel moved". Tuned for
# darts' 50ms FLIGHT_POLL — fast-moving dart pixels easily exceed 25,
# while sweeping arm pixels usually stay under.
_MOTION_THRESHOLD = 25


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def find_dart_in_flight(
    cur: np.ndarray,
    prev: np.ndarray | None,
) -> tuple[int, int] | None:
    """Detect the dart's centroid (x, y) in `cur`, requiring motion vs.
    `prev`. Returns None if no dart-shaped blob is found.

    Picks the largest-area gray+moving connected component. Ties are
    broken by component area; if the player's helmet is briefly
    flagged as moving (rare under normal sweep speeds) it'll usually
    be larger than the dart and dominate — but the held-dart edge
    case is the more common false positive, and it's filtered by
    being mostly stationary.
    """
    cur_bgr = _ensure_bgr(cur)
    cur_hsv = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2HSV)
    gray_mask = (cur_hsv[..., 1] < _GRAY_SAT_MAX) & (cur_hsv[..., 2] > _GRAY_VAL_MIN)

    if prev is not None:
        prev_bgr = _ensure_bgr(prev)
        diff = cv2.absdiff(cur_bgr, prev_bgr).max(axis=2)
        motion_mask = diff > _MOTION_THRESHOLD
        mask = gray_mask & motion_mask
    else:
        mask = gray_mask

    mask_u8 = mask.astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    best: tuple[int, int, int] | None = None  # (cx, cy, area)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (_DART_AREA_MIN <= area <= _DART_AREA_MAX):
            continue
        # Prefer horizontally-elongated components (the dart shaft).
        if h > w * 2:
            continue
        cx = x + w // 2
        cy = y + h // 2
        if best is None or area > best[2]:
            best = (cx, cy, int(area))

    if best is None:
        return None
    return best[0], best[1]


def track_dart_trajectory(
    frames: list[np.ndarray],
) -> list[tuple[int, int, int]]:
    """Run the dart finder across consecutive frames and return
    (frame_idx, x, y) for frames where the dart was detected. frame_idx
    is 1-based to match the flight_NNN.png filename convention.
    """
    positions: list[tuple[int, int, int]] = []
    for i in range(1, len(frames)):
        pos = find_dart_in_flight(frames[i], frames[i - 1])
        if pos is not None:
            positions.append((i + 1, pos[0], pos[1]))
    return positions


def summarise_trajectory(
    positions: list[tuple[int, int, int]],
    early_frame_count: int = 3,
) -> dict:
    """Reduce a (frame_idx, x, y) trajectory to summary metrics.

    launch_angle is computed from the first up-to-`early_frame_count`
    detected positions: a least-squares fit through them gives the
    velocity vector, and angle = atan2(dy, dx) in degrees. Angle 0
    points right; positive angles point downward (screen coords).

    Returns dict with keys: launch_angle_deg, apex_y, landing_x,
    frames_seen. All None if positions is empty (apart from
    frames_seen which is 0).
    """
    if not positions:
        return {
            "launch_angle_deg": None,
            "apex_y": None,
            "landing_x": None,
            "frames_seen": 0,
        }
    apex_y = min(p[2] for p in positions)
    landing_x = positions[-1][1]

    angle_deg: float | None = None
    if len(positions) >= 2:
        early = positions[:early_frame_count]
        xs = np.array([p[1] for p in early], dtype=np.float64)
        ys = np.array([p[2] for p in early], dtype=np.float64)
        if len(early) == 2:
            dx = xs[1] - xs[0]
            dy = ys[1] - ys[0]
        else:
            # Least-squares slope through (xs, ys); if dx is small,
            # fall back to first-vs-last difference to avoid div-by-zero.
            dx = xs[-1] - xs[0]
            dy = ys[-1] - ys[0]
        if dx != 0 or dy != 0:
            angle_deg = float(np.degrees(np.arctan2(dy, dx)))

    return {
        "launch_angle_deg": angle_deg,
        "apex_y": int(apex_y),
        "landing_x": int(landing_x),
        "frames_seen": len(positions),
    }


def analyse_throw_dir(throw_dir: Path) -> dict:
    """Load flight_*.png from throw_dir, track the dart, and summarise.
    Returns the same keys as summarise_trajectory; all-None values
    (and frames_seen=0) if no frames present or none detected."""
    frame_paths = sorted(throw_dir.glob("flight_*.png"))
    if not frame_paths:
        return summarise_trajectory([])
    frames = [cv2.imread(str(p)) for p in frame_paths]
    frames = [f for f in frames if f is not None]
    positions = track_dart_trajectory(frames)
    return summarise_trajectory(positions)
