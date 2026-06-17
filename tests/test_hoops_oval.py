"""Tests for the hoops 20-make oval-phase helpers (#59) — pure logic."""
from minigames.hoops.main import (
    OVAL_X_RANGE_PX,
    _platform_velocity_x,
    _platform_x_range,
    _platform_recently_moving,
    _oval_active,
)


def _samples(px_fn, py=400, n=11, dt=0.1):
    """(timestamp, px, py) buffer with px = px_fn(i)."""
    return [(i * dt, px_fn(i), py) for i in range(n)]


def test_platform_velocity_x_recovers_rightward_slope():
    # px = 100 + 200*t  -> +200 px/s rightward.
    s = [(t / 10.0, int(100 + 200 * (t / 10.0)), 400) for t in range(11)]
    vx = _platform_velocity_x(s)
    assert vx is not None and abs(vx - 200) < 5


def test_platform_velocity_x_none_when_too_short():
    assert _platform_velocity_x([]) is None
    assert _platform_velocity_x([(0.0, 100, 400)]) is None
    # span < min_span_s (0.2)
    assert _platform_velocity_x([(0.0, 100, 400), (0.05, 110, 400)]) is None


def test_platform_x_range():
    assert _platform_x_range([]) == 0
    assert _platform_x_range(_samples(lambda i: 136)) == 0
    assert _platform_x_range(_samples(lambda i: 100 + 10 * i)) == 100  # i=0..10


def test_oval_inactive_below_min_samples():
    # Wide x-spread but only 11 samples -> not enough to trust.
    assert _oval_active(_samples(lambda i: 86 + 10 * i, n=11)) is False


def test_oval_inactive_when_fixed_x():
    # 50 samples, x locked (vertical bob) -> not the oval.
    assert _oval_active(_samples(lambda i: 136, n=50)) is False


def test_oval_active_when_x_spread_opens():
    # 50 samples spanning ~100px (oval) -> active.
    s = _samples(lambda i: 86 + (i % 11) * 10, n=50)
    assert _platform_x_range(s) > OVAL_X_RANGE_PX
    assert _oval_active(s) is True


def test_oval_robust_to_single_x_glitch():
    # 49 samples locked at x=136 + one detector glitch to 374: raw max-min
    # blows past the threshold, but the percentile spread ignores the lone
    # outlier, so oval-fire mode must NOT trip during normal play.
    s = _samples(lambda i: 136, n=49) + [(4.9, 374, 400)]
    assert _platform_x_range(s) > OVAL_X_RANGE_PX     # raw range is fooled
    assert _oval_active(s) is False                   # percentile spread isn't


def test_oval_drops_when_platform_stops_moving():
    # Undetected game-over: buffer still holds a wide oval spread (first 35
    # samples), but the platform is now static (last 15 identical). oval_active
    # must drop so the runaway bail can arm.
    moving = [(i * 0.1, 86 + (i % 11) * 10, 400) for i in range(35)]
    stuck = [((35 + i) * 0.1, 150, 400) for i in range(15)]
    s = moving + stuck
    xs = sorted(p[1] for p in s)
    assert xs[int(len(s) * 0.9)] - xs[int(len(s) * 0.1)] > OVAL_X_RANGE_PX  # buffer still wide
    assert _platform_recently_moving(s) is False
    assert _oval_active(s) is False


def test_recently_moving_true_during_live_oval():
    # Continuous oval motion -> recent window always shows movement.
    s = _samples(lambda i: 86 + (i % 11) * 10, n=50)
    assert _platform_recently_moving(s) is True
