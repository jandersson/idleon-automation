"""Tests for common.dart_trajectory — synthetic frames only, no live capture."""
import numpy as np

from common.dart_trajectory import (
    find_dart_in_flight, track_dart_trajectory, summarise_trajectory,
)


def _frame_with_dart(width: int, height: int, x: int, y: int) -> np.ndarray:
    """Create a BGR frame with a small gray dart-shaped sliver at (x, y).

    The detector requires gray (low-sat, mid-high value) + horizontally
    elongated. Painting a 12×3 light-gray rectangle with a darker tip
    matches both criteria.
    """
    bg = np.full((height, width, 3), (40, 80, 120), dtype=np.uint8)  # orange-brown wall, high sat
    bg[y:y+3, x:x+10] = (200, 200, 200)  # gray shaft
    bg[y:y+3, x+10:x+12] = (180, 180, 180)  # slightly darker tip
    return bg


def test_find_dart_with_motion_returns_centroid():
    """The dart must move between frames to be detected — pure-static
    gray pixels are filtered by the motion mask."""
    prev = _frame_with_dart(200, 100, 30, 50)
    cur = _frame_with_dart(200, 100, 80, 48)  # moved right and slightly up
    pos = find_dart_in_flight(cur, prev)
    assert pos is not None
    cx, cy = pos
    # Centroid of the 12×3 sliver at (80, 48): roughly (86, 49).
    assert 80 <= cx <= 96
    assert 47 <= cy <= 51


def test_find_dart_returns_none_when_static():
    """If the dart didn't move (or no prev frame), motion mask kills
    every gray pixel — detector returns None."""
    f = _frame_with_dart(200, 100, 30, 50)
    pos = find_dart_in_flight(f, f)  # zero motion
    assert pos is None


def test_track_dart_trajectory_returns_progression():
    """Three frames with the dart marching right and slightly down →
    we should detect motion in frames 2 and 3 (1-indexed → frame_idx
    2 and 3)."""
    frames = [
        _frame_with_dart(300, 100, 20, 50),
        _frame_with_dart(300, 100, 60, 53),
        _frame_with_dart(300, 100, 100, 57),
    ]
    positions = track_dart_trajectory(frames)
    assert len(positions) == 2
    idx1, x1, _ = positions[0]
    idx2, x2, _ = positions[1]
    assert idx1 == 2 and idx2 == 3
    assert x2 > x1  # moving right


def test_summarise_trajectory_computes_launch_angle():
    """A pure-rightward dart at zero vertical change → angle = 0°.
    A right-and-down dart → positive angle (screen y increases down)."""
    pure_right = [(2, 50, 100), (3, 100, 100), (4, 150, 100)]
    res = summarise_trajectory(pure_right)
    assert abs(res["launch_angle_deg"] - 0.0) < 1e-6

    right_and_down = [(2, 50, 100), (3, 100, 110)]
    res = summarise_trajectory(right_and_down)
    assert res["launch_angle_deg"] > 0


def test_summarise_trajectory_handles_empty():
    res = summarise_trajectory([])
    assert res == {
        "launch_angle_deg": None,
        "apex_y": None,
        "landing_x": None,
        "frames_seen": 0,
    }


def test_summarise_trajectory_picks_apex_and_landing():
    """apex_y is min y observed; landing_x is x of the last position."""
    positions = [(2, 50, 200), (3, 100, 150), (4, 150, 100), (5, 200, 130), (6, 250, 180)]
    res = summarise_trajectory(positions)
    assert res["apex_y"] == 100
    assert res["landing_x"] == 250
    assert res["frames_seen"] == 5
