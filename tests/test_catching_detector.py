"""Tests for minigames.catching.detector.find_next_gap (#47).

find_next_gap returns the next gold ring's full bounding box
(top_y, bottom_y, left_x, right_x) — the horizontal extent is logged so
analysis can correlate flap timing with how far ahead the ring is.
"""
import numpy as np

from minigames.catching.detector import find_next_gap


def _gold_ring_frame() -> np.ndarray:
    """A BGRA frame with one gold square (in the ring HSV band) at
    rows 30:70, cols 100:140 — large enough to clear RING_MIN_AREA."""
    frame = np.zeros((100, 200, 4), dtype=np.uint8)
    frame[30:70, 100:140] = (0, 200, 230, 255)  # B,G,R,A — saturated gold/orange
    return frame


def test_find_next_gap_returns_full_bbox_ahead_of_fly():
    frame = _gold_ring_frame()
    gap = find_next_gap(frame, fly_pos=(10, 50))  # fly to the left of the ring
    assert gap is not None
    top_y, bottom_y, left_x, right_x = gap  # must be a 4-tuple now
    assert (top_y, bottom_y, left_x, right_x) == (30, 70, 100, 140)


def test_find_next_gap_ignores_rings_already_passed():
    """A ring whose center is left of the fly has been passed — skip it."""
    frame = _gold_ring_frame()  # ring center x = 120
    assert find_next_gap(frame, fly_pos=(180, 50)) is None


def test_find_next_gap_none_without_fly():
    assert find_next_gap(_gold_ring_frame(), fly_pos=None) is None
