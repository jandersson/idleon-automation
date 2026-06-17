"""Tests for the fishing cast model + target selection (pure logic)."""
from minigames.fishing.cast_model import (
    CastModel,
    MIN_SAMPLES,
    fit_cast_model,
    distance_for_hold,
    hold_for_distance,
    reachable,
    choose_target,
)


def _linear_samples(slope=2.0, intercept=50.0, n=MIN_SAMPLES):
    return [(h, slope * h + intercept) for h in range(100, 100 + 10 * n, 10)]


def test_fit_recovers_linear_relationship():
    m = fit_cast_model(_linear_samples(slope=2.0, intercept=50.0))
    assert m is not None
    assert abs(m.slope - 2.0) < 1e-6
    assert abs(m.intercept - 50.0) < 1e-6
    assert m.hold_min_ms == 100
    assert m.n >= MIN_SAMPLES


def test_fit_none_below_min_samples():
    assert fit_cast_model(_linear_samples(n=MIN_SAMPLES - 1)) is None


def test_fit_none_on_nonpositive_slope():
    # Holding longer must cast farther; a flat/decreasing fit is noise.
    assert fit_cast_model([(h, 500.0) for h in range(100, 100 + 10 * MIN_SAMPLES, 10)]) is None
    assert fit_cast_model([(h, -2.0 * h) for h in range(100, 100 + 10 * MIN_SAMPLES, 10)]) is None


def test_fit_none_on_identical_holds():
    assert fit_cast_model([(300, float(d)) for d in range(MIN_SAMPLES + 5)]) is None


def test_hold_for_distance_inverts_and_clamps():
    m = CastModel(slope=2.0, intercept=50.0, hold_min_ms=100, hold_max_ms=1000, n=20)
    assert hold_for_distance(m, 450) == 200          # (450-50)/2
    assert distance_for_hold(m, 200) == 450
    assert hold_for_distance(m, 100) == 100          # below reach -> clamp to hold_min
    assert hold_for_distance(m, 99_999) == 1000      # above reach -> clamp to hold_max


def test_reachable():
    m = CastModel(slope=2.0, intercept=50.0, hold_min_ms=100, hold_max_ms=1000, n=20)
    # reach = [250, 2050]
    assert reachable(m, 400)
    assert not reachable(m, 100)
    assert not reachable(m, 5000)
    assert reachable(None, 99_999)  # no model -> exploration reaches by trying


def test_choose_target_prefers_value_then_nearest():
    m = CastModel(slope=2.0, intercept=50.0, hold_min_ms=100, hold_max_ms=1000, n=20)
    origin = 500
    fish = [
        {"x": 900, "kind": "green"},   # dist 400, value 1
        {"x": 700, "kind": "squid"},   # dist 200 (reachable? 200<250 -> NO)
        {"x": 800, "kind": "eel"},     # dist 300, value 2 -> reachable, higher value
        {"x": 3000, "kind": "whale"},  # dist 2500 -> unreachable (>2050)
    ]
    chosen = choose_target(fish, m, origin)
    assert chosen is not None
    assert chosen["kind"] == "eel" and chosen["target_dist"] == 300 and chosen["value"] == 2


def test_choose_target_tie_break_nearest_value():
    m = CastModel(slope=1.0, intercept=0.0, hold_min_ms=0, hold_max_ms=5000, n=20)
    origin = 0
    fish = [
        {"x": 800, "kind": "green"},
        {"x": 400, "kind": "green"},   # same value, nearer -> chosen
    ]
    chosen = choose_target(fish, m, origin)
    assert chosen["target_dist"] == 400


def test_choose_target_none_when_unreachable_or_empty():
    m = CastModel(slope=2.0, intercept=50.0, hold_min_ms=100, hold_max_ms=1000, n=20)
    assert choose_target([], m, 0) is None
    assert choose_target([{"x": 10_000, "kind": "green"}], m, 0) is None


def test_choose_target_with_no_model_takes_everything():
    fish = [{"x": 9_999, "kind": "whale"}, {"x": 10, "kind": "green"}]
    chosen = choose_target(fish, None, 0)
    assert chosen["kind"] == "whale"  # highest value, reachable since no model
