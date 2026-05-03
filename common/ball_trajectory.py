"""Extract ball trajectory metrics from per-shot flight frames.

The hoops bot already captures flight_*.png at 50ms intervals during the
post-shot cooldown. Re-running the orange-ball HSV+motion detector across
those frames gives us a (frame_idx, x, y) trajectory, which we summarise
into three metrics:

- ball_apex_y       : the highest point the ball reached (smallest y)
- ball_x_at_rim_height : ball x when its y was closest to hoop_y
                       (None if the ball never came near rim height)
- ball_landing_x    : x of the last detected position before the ball
                       leaves the visible area

These let us classify misses automatically:
- ball_x_at_rim_height < hoop_x → undershoot (front rim or short)
- ball_x_at_rim_height > hoop_x → overshoot (back rim or sailed past)
- big abs(apex_y - hoop_y) → arc shape was wrong
"""
import sys
from pathlib import Path

import cv2
import numpy as np

# detector.find_ball needs the same HSV bounds and motion masking, plus
# search bounds; we pass full-frame bounds here.
sys.path.insert(0, str(Path(__file__).parent.parent))
from minigames.hoops.detector import find_ball


def track_ball_trajectory(
    frames: list[np.ndarray],
    search_x0: int = 200,
    motion_threshold: float = 12.0,
) -> list[tuple[int, int, int]]:
    """Find the ball in each frame (using motion-masked HSV detection against
    the previous frame). Returns a list of (frame_idx, x, y) for frames where
    the ball was detected. frame_idx is 1-based to match flight_NNN naming.

    search_x0 cuts off the left edge so the player's sprite (around x=136
    in the 960-wide window) doesn't show up as a moving orange blob and
    get picked up as the ball.
    """
    if not frames:
        return []
    h, w = frames[0].shape[:2]
    positions: list[tuple[int, int, int]] = []
    for i in range(1, len(frames)):
        # find_ball expects BGRA. Caller passes whatever loader gives us;
        # if it's BGR (3-channel from cv2.imread), convert to BGRA so
        # detector's COLOR_BGRA2BGR step works correctly.
        cur = _ensure_bgra(frames[i])
        prev = _ensure_bgra(frames[i - 1])
        ball = find_ball(cur, search_x0, 0, w, h, prev_frame=prev,
                         motion_threshold=motion_threshold)
        if ball is not None:
            positions.append((i + 1, ball[0], ball[1]))  # 1-based
    return positions


def _ensure_bgra(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def summarise_trajectory(
    positions: list[tuple[int, int, int]],
    hoop_x: int,
    hoop_y: int,
    rim_height_tolerance: int = 25,
) -> dict:
    """Reduce a (frame_idx, x, y) trajectory to summary metrics.

    Returns a dict with keys:
        ball_apex_y           : min(y) observed, or None
        ball_x_at_rim_height  : x at the point where y was closest to hoop_y
                                (only if within rim_height_tolerance px)
        ball_landing_x        : x of the last detected position
    All values are None if the trajectory is empty.
    """
    if not positions:
        return {
            "ball_apex_y": None,
            "ball_x_at_rim_height": None,
            "ball_landing_x": None,
        }
    apex_y = min(p[2] for p in positions)
    landing_x = positions[-1][1]
    # Closest position to rim height (smallest abs y diff).
    closest = min(positions, key=lambda p: abs(p[2] - hoop_y))
    x_at_rim = closest[1] if abs(closest[2] - hoop_y) <= rim_height_tolerance else None
    return {
        "ball_apex_y": int(apex_y),
        "ball_x_at_rim_height": int(x_at_rim) if x_at_rim is not None else None,
        "ball_landing_x": int(landing_x),
    }


def analyse_shot_dir(shot_dir: Path, hoop_x: int, hoop_y: int) -> dict:
    """Load flight_*.png from shot_dir, track and summarise. Returns the
    same keys as summarise_trajectory; all-None if no frames present."""
    frame_paths = sorted(shot_dir.glob("flight_*.png"))
    if not frame_paths:
        return summarise_trajectory([], hoop_x, hoop_y)
    frames = [cv2.imread(str(p)) for p in frame_paths]
    frames = [f for f in frames if f is not None]
    positions = track_ball_trajectory(frames)
    return summarise_trajectory(positions, hoop_x, hoop_y)
