"""Tests for the fishing cast model + target selection (pure logic).

The model is charge_level -> landing-distance (the closed-loop cast releases
at a target charge fill); the distance cap falls out of the charge support.
"""
from minigames.fishing.cast_model import (
    CastModel,
    MIN_SAMPLES,
    fit_cast_model,
    distance_for_charge,
    charge_for_distance,
    reachable,
    choose_target,
    nearest_fish_target,
    lands_on_mine_only,
)


def _linear_samples(slope=5.0, intercept=-20.0, n=MIN_SAMPLES):
    # charge fills span ~10..60 live; sample that range
    return [(c, slope * c + intercept) for c in range(10, 10 + 4 * n, 4)]


def test_fit_recovers_linear_relationship():
    m = fit_cast_model(_linear_samples(slope=5.0, intercept=-20.0))
    assert m is not None
    assert abs(m.slope - 5.0) < 1e-6
    assert abs(m.intercept - (-20.0)) < 1e-6
    assert m.charge_min == 10
    assert m.n >= MIN_SAMPLES


def test_fit_is_robust_to_landing_outliers():
    # The bobber landing (the y) is still noisy even though charge (x) is clean:
    # an off-crop / mis-latched landing gives an outlier. Theil-Sen recovers the
    # true slope despite a couple of those.
    pts = _linear_samples(slope=5.0, intercept=-20.0, n=MIN_SAMPLES)
    pts[3] = (pts[3][0], 5.0)      # clean charge, garbage landing
    pts[7] = (pts[7][0], 8.0)
    m = fit_cast_model(pts)
    assert m is not None
    assert abs(m.slope - 5.0) < 0.3   # least-squares would be dragged well off


def test_fit_none_below_min_samples():
    assert fit_cast_model(_linear_samples(n=MIN_SAMPLES - 1)) is None


def test_fit_none_on_nonpositive_slope():
    # More charge must cast farther; a flat/decreasing fit is noise.
    assert fit_cast_model([(c, 200.0) for c in range(10, 10 + 4 * MIN_SAMPLES, 4)]) is None
    assert fit_cast_model([(c, -5.0 * c) for c in range(10, 10 + 4 * MIN_SAMPLES, 4)]) is None


def test_fit_none_on_identical_charges():
    assert fit_cast_model([(40, float(d)) for d in range(MIN_SAMPLES + 5)]) is None


def test_charge_for_distance_inverts_and_clamps():
    m = CastModel(slope=5.0, intercept=0.0, charge_min=10, charge_max=60, n=20)
    assert charge_for_distance(m, 200) == 40          # 200/5
    assert distance_for_charge(m, 40) == 200
    assert charge_for_distance(m, 10) == 10           # below reach -> clamp to charge_min
    assert charge_for_distance(m, 99_999) == 60       # above reach (bar saturated) -> charge_max


def test_charge_for_distance_min_charge_override_extrapolates_down():
    # The close-fish 'nearest' aim reaches BELOW the sampled support (charge_min)
    # toward a fish nearer than the reach floor, down to the given floor.
    m = CastModel(slope=5.0, intercept=0.0, charge_min=10, charge_max=60, n=20)
    assert charge_for_distance(m, 50) == 10                      # default floor = charge_min
    assert charge_for_distance(m, 50, min_charge=5) == 10        # 50/5 -> 10, above the 5 floor
    assert charge_for_distance(m, 20, min_charge=5) == 5         # 20/5=4 -> floored at 5 (was 10)
    assert charge_for_distance(m, 99_999, min_charge=5) == 60    # upper clamp unaffected


def test_nearest_fish_target():
    assert nearest_fish_target([], origin_x=0) is None
    fish = [{"x": 200, "kind": "squid"}, {"x": 50, "kind": "green"}]
    t = nearest_fish_target(fish, origin_x=0)
    assert t["x"] == 50 and t["kind"] == "green"        # nearest to the origin
    assert t["target_dist"] == 50 and t["value"] == 1   # augmented dist + point value


def test_reachable():
    m = CastModel(slope=5.0, intercept=0.0, charge_min=10, charge_max=60, n=20)
    # reach = [50, 300]
    assert reachable(m, 200)
    assert not reachable(m, 40)       # nearer than the min charge can place it
    assert not reachable(m, 400)      # past the bar's saturation
    assert reachable(None, 99_999)    # no model -> exploration reaches by trying


def test_choose_target_prefers_value_then_nearest():
    m = CastModel(slope=5.0, intercept=0.0, charge_min=10, charge_max=60, n=20)
    origin = 0
    # reach = [50, 300] px from origin
    fish = [
        {"x": 250, "kind": "green"},   # dist 250, value 1 (reachable)
        {"x": 40, "kind": "squid"},    # dist 40 -> < 50, NOT reachable
        {"x": 200, "kind": "eel"},     # dist 200, value 2 -> reachable, higher value
        {"x": 900, "kind": "whale"},   # dist 900 -> unreachable (>300)
    ]
    chosen = choose_target(fish, m, origin)
    assert chosen is not None
    assert chosen["kind"] == "eel" and chosen["target_dist"] == 200 and chosen["value"] == 2


def test_choose_target_tie_break_nearest_value():
    m = CastModel(slope=1.0, intercept=0.0, charge_min=0, charge_max=5000, n=20)
    origin = 0
    fish = [
        {"x": 800, "kind": "green"},
        {"x": 400, "kind": "green"},   # same value, nearer -> chosen
    ]
    chosen = choose_target(fish, m, origin)
    assert chosen["target_dist"] == 400


def test_choose_target_none_when_unreachable_or_empty():
    m = CastModel(slope=5.0, intercept=0.0, charge_min=10, charge_max=60, n=20)
    assert choose_target([], m, 0) is None
    assert choose_target([{"x": 10_000, "kind": "green"}], m, 0) is None


def test_choose_target_with_no_model_takes_everything():
    fish = [{"x": 9_999, "kind": "whale"}, {"x": 10, "kind": "green"}]
    chosen = choose_target(fish, None, 0)
    assert chosen["kind"] == "whale"  # highest value, reachable since no model


def test_lands_on_mine_only():
    mines = [{"x": 200}]
    # mine with no fish -> fail spot
    assert lands_on_mine_only(202, mines, fish=[], tol_px=15)
    # a fish over the mine -> scores, not a fail spot
    assert not lands_on_mine_only(202, mines, fish=[{"x": 200, "kind": "green"}], tol_px=15)
    # open water (no mine near) -> fine
    assert not lands_on_mine_only(100, mines, fish=[], tol_px=15)
    # mine too far from the landing -> fine
    assert not lands_on_mine_only(250, mines, fish=[], tol_px=15)
