"""Tests for hoops _compute_offset (linear interpolation between anchors)."""
from minigames.hoops.main import OFFSET_ANCHORS, _compute_offset


def test_endpoints_match_anchors():
    for hoop_y, expected_offset in OFFSET_ANCHORS:
        assert _compute_offset(hoop_y) == expected_offset


def test_clamps_below_first_anchor():
    first_y, first_offset = sorted(OFFSET_ANCHORS)[0]
    assert _compute_offset(first_y - 100) == first_offset
    assert _compute_offset(0) == first_offset


def test_clamps_above_last_anchor():
    last_y, last_offset = sorted(OFFSET_ANCHORS)[-1]
    assert _compute_offset(last_y + 100) == last_offset


def test_interpolates_between_anchors():
    pts = sorted(OFFSET_ANCHORS)
    # Pick the first interval and check the midpoint.
    (y1, o1), (y2, o2) = pts[0], pts[1]
    mid = (y1 + y2) // 2
    expected = round((o1 + o2) / 2)
    # Allow ±1 for int rounding.
    assert abs(_compute_offset(mid) - expected) <= 1


def test_increases_or_decreases_monotonically_within_interval():
    """The output should track the anchor slope (no weird non-monotonicity)."""
    pts = sorted(OFFSET_ANCHORS)
    if len(pts) < 2:
        return
    (y1, o1), (y2, o2) = pts[0], pts[1]
    samples = [_compute_offset(y1 + step) for step in range(0, y2 - y1 + 1, max(1, (y2 - y1) // 10))]
    if o1 < o2:
        assert all(a <= b for a, b in zip(samples, samples[1:]))
    elif o1 > o2:
        assert all(a >= b for a, b in zip(samples, samples[1:]))
    else:
        assert all(s == samples[0] for s in samples)
