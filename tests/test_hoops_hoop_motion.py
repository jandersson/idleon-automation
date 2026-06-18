"""Tests for the hoops hoop horizontal-motion diagnostics (#76) — pure logic.

The hoop oscillates in x but the aim locks hoop_x once at target-set, so a
moving hoop is stale by the time the ball arrives ~1.5s later. _hoop_motion
summarises a throttled rim buffer per shot (hoop_vx, oscillation range,
freshest x) — instrumentation only; the aim path is unchanged.
"""
from minigames.hoops.main import _hoop_motion


def _buf(xs, dt=0.1, y=300):
    """(timestamp, hoop_x, hoop_y) buffer from a list of hoop_x values."""
    return [(i * dt, x, y) for i, x in enumerate(xs)]


def test_static_hoop_zero_velocity_single_x():
    vx, xmin, xmax, x_at_fire = _hoop_motion(_buf([640] * 10))
    assert vx is not None and abs(vx) < 1
    assert xmin == xmax == 640
    assert x_at_fire == 640


def test_rightward_drift_positive_vx_and_range():
    # hoop_x = 600 + 10*i over 1s (dt=0.1) -> +100 px/s, range 600..700.
    vx, xmin, xmax, x_at_fire = _hoop_motion(_buf([600 + 10 * i for i in range(11)]))
    assert vx is not None and abs(vx - 100) < 5
    assert (xmin, xmax) == (600, 700)
    assert x_at_fire == 700  # the freshest sample, not the target-set one


def test_oscillation_range_captured_and_freshest_x_is_last():
    # Up to 700 then back to 620: the range spans the full swing, and
    # x_at_fire is the latest position — the staleness vs the logged
    # (target-set) hoop_x is exactly what this instruments.
    xs = [640, 660, 680, 700, 680, 660, 640, 620]
    _, xmin, xmax, x_at_fire = _hoop_motion(_buf(xs))
    assert (xmin, xmax) == (620, 700)
    assert x_at_fire == 620


def test_empty_buffer_all_none():
    assert _hoop_motion([]) == (None, None, None, None)


def test_under_five_samples_no_range_but_keeps_freshest_x():
    # <5 samples: don't trust a range, but the freshest x is still useful.
    _, xmin, xmax, x_at_fire = _hoop_motion(_buf([630, 645], dt=0.3))
    assert xmin is None and xmax is None
    assert x_at_fire == 645
