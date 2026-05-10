"""Tests for common.predictor — the swappable Predictor implementations."""
import pytest

from common.predictor import (
    KnnPredictor, BivariatePredictor, TrajectoryKnnPredictor, TrajectoryGpPredictor,
    fit_knn, fit_bivariate, fit_gp, fit_trajectory_knn, fit_trajectory_gp,
)


def test_fit_knn_returns_none_with_too_few_samples():
    rows = [(400.0, 700.0, 420.0), (410.0, 700.0, 430.0), (420.0, 700.0, 440.0)]
    assert fit_knn(rows, min_samples=4) is None


def test_fit_knn_clamps_k_to_dataset_size():
    rows = [(400.0, 700.0, 420.0)] * 4
    pred = fit_knn(rows, k=10)
    assert pred is not None
    assert pred.k == 4  # clamped


def test_knn_predicts_exact_match_via_inverse_distance():
    points = [
        (400.0, 700.0, 420.0),
        (300.0, 600.0, 320.0),
        (500.0, 800.0, 520.0),
        (350.0, 650.0, 370.0),
    ]
    pred = KnnPredictor(points, k=1)
    assert pred.predict(400, 700) == 420  # exact match wins outright


def test_knn_blends_neighbours():
    """Equidistant query: with k=2 returns weighted average."""
    points = [(400.0, 600.0, 400.0), (400.0, 800.0, 440.0)]
    pred = KnnPredictor(points, k=2)
    # at (400, 700) both points are 100 away — average of 400 and 440 = 420
    assert pred.predict(400, 700) == 420


def test_fit_bivariate_recovers_known_plane():
    """target_y = 0.5*hoop_y + 0.3*hoop_x + 10."""
    rows = [
        (300.0, 600.0, 0.5*300 + 0.3*600 + 10),
        (300.0, 700.0, 0.5*300 + 0.3*700 + 10),
        (400.0, 600.0, 0.5*400 + 0.3*600 + 10),
        (400.0, 700.0, 0.5*400 + 0.3*700 + 10),
    ]
    pred = fit_bivariate(rows)
    assert pred is not None
    assert pred.n == 4
    assert abs(pred.a - 0.5) < 1e-6
    assert abs(pred.b - 0.3) < 1e-6
    assert abs(pred.c - 10.0) < 1e-6
    # Predicts on a held-out point.
    expected = 0.5 * 350 + 0.3 * 650 + 10
    assert abs(pred.predict(350, 650) - expected) < 1e-6


def test_fit_bivariate_returns_none_on_singular_input():
    """Perfectly collinear points (hoop_y == hoop_x) give a singular
    normal matrix; fit returns None instead of crashing."""
    rows = [(300.0, 300.0, 320.0), (400.0, 400.0, 420.0),
            (500.0, 500.0, 520.0), (600.0, 600.0, 620.0)]
    assert fit_bivariate(rows) is None


def test_fit_bivariate_returns_none_with_too_few_samples():
    rows = [(300.0, 600.0, 320.0), (400.0, 700.0, 420.0)]
    assert fit_bivariate(rows, min_samples=4) is None


def test_predictors_have_compatible_interface():
    """Both implementations expose .n and .predict(hoop_y, hoop_x).
    Use non-collinear points so the bivariate fit is well-conditioned."""
    rows = [
        (300.0, 600.0, 320.0), (300.0, 700.0, 330.0),
        (400.0, 600.0, 420.0), (400.0, 700.0, 430.0),
    ]
    knn = fit_knn(rows)
    bv = fit_bivariate(rows)
    assert knn is not None and bv is not None
    for p in (knn, bv):
        assert isinstance(p.n, int)
        assert isinstance(p.predict(400, 700), float)


def test_fit_gp_returns_none_with_too_few_samples():
    pytest.importorskip("sklearn")
    rows = [(400.0, 700.0, 420.0), (410.0, 700.0, 430.0), (420.0, 700.0, 440.0)]
    assert fit_gp(rows, min_samples=4) is None


def test_gp_fits_and_returns_mean_near_training_target():
    """At a training input the GP posterior mean should be close to the
    observed y (modulo the noise term in the WhiteKernel)."""
    pytest.importorskip("sklearn")
    rows = [
        (300.0, 600.0, 320.0), (300.0, 700.0, 330.0),
        (400.0, 600.0, 420.0), (400.0, 700.0, 430.0),
        (500.0, 600.0, 520.0), (500.0, 700.0, 530.0),
    ]
    pred = fit_gp(rows)
    assert pred is not None
    assert pred.n == 6
    # Within a few px of the observed y at a training input.
    assert abs(pred.predict(400, 700) - 430) < 5


def test_gp_predict_with_std_returns_two_floats():
    """predict_with_std yields (mean, std); std is non-negative and rises
    away from the training cluster (the property the bot will exploit).

    Use a denser training cluster so the GP actually learns structure
    instead of saturating on a 4-point prior. Query points are picked
    well inside vs. well outside the cluster's convex hull.
    """
    pytest.importorskip("sklearn")
    rows = [
        (350.0, 620.0, 370.0), (350.0, 660.0, 375.0), (350.0, 700.0, 380.0),
        (400.0, 620.0, 420.0), (400.0, 660.0, 425.0), (400.0, 700.0, 430.0),
        (450.0, 620.0, 470.0), (450.0, 660.0, 475.0), (450.0, 700.0, 480.0),
    ]
    pred = fit_gp(rows)
    assert pred is not None
    mu_in, std_in = pred.predict_with_std(400, 660)  # smack inside the grid
    mu_out, std_out = pred.predict_with_std(900, 200)  # far outside
    assert isinstance(mu_in, float) and isinstance(std_in, float)
    assert std_in >= 0 and std_out >= 0
    # Far from training data → larger uncertainty.
    assert std_out > std_in


def test_gp_matches_predictor_protocol():
    """Same .n / .predict surface as the other predictors."""
    pytest.importorskip("sklearn")
    rows = [
        (300.0, 600.0, 320.0), (300.0, 700.0, 330.0),
        (400.0, 600.0, 420.0), (400.0, 700.0, 430.0),
    ]
    gp = fit_gp(rows)
    assert gp is not None
    assert isinstance(gp.n, int)
    assert isinstance(gp.predict(400, 700), float)


def test_fit_trajectory_knn_returns_none_with_too_few_samples():
    rows = [(400.0, 700.0, 420.0, 700.0)] * 3
    assert fit_trajectory_knn(rows, min_samples=8) is None


def _synthetic_trajectory_rows() -> list[tuple[float, float, float, float]]:
    """Build a synthetic dataset where landing_x is a clean function of
    (hoop_y, hoop_x, platform_y). The optimal platform_y for a given
    hoop is the one whose modelled landing equals hoop_x.

    Model: each hoop_y has its own offset slope; landing_x = hoop_x +
    slope * (platform_y - 420). So landing == hoop_x exactly when
    platform_y == 420 (regardless of hoop_y / hoop_x for this fixture).
    """
    rows: list[tuple[float, float, float, float]] = []
    for hoop_y in (350.0, 400.0, 450.0):
        for hoop_x in (600.0, 700.0):
            slope = 0.5  # px landing per px platform_y above 420
            for platform_y in (390.0, 410.0, 420.0, 430.0, 450.0):
                landing_x = hoop_x + slope * (platform_y - 420.0)
                rows.append((hoop_y, hoop_x, platform_y, landing_x))
    return rows


def test_fit_trajectory_knn_returns_predictor_when_enough_samples():
    rows = _synthetic_trajectory_rows()
    pred = fit_trajectory_knn(rows, k=3)
    assert pred is not None
    assert pred.n == len(rows)


def test_trajectory_knn_recovers_optimal_platform_y():
    """For the synthetic fixture the optimum is always platform_y=420
    (the launch elevation that yields landing_x == hoop_x). At a query
    point dense in training data, 3-NN locates the optimum within one
    scan step."""
    rows = _synthetic_trajectory_rows()
    pred = fit_trajectory_knn(rows, k=3, py_scan_step=5.0)
    assert pred is not None
    # Query at a trained (hoop_y, hoop_x) so 3-NN finds neighbors with
    # zero spatial offset and the platform_y axis dominates the distance
    # metric — avoids cross-column averaging artefacts.
    py_pred = pred.predict(400, 600)
    assert abs(py_pred - 420.0) <= 5.0


def test_trajectory_knn_scan_bounded_by_training_range():
    """Scan should not return platform_y values outside the training set's
    [min, max] envelope — extrapolation past those bounds is not allowed.
    Backward-compat: when no makes are provided, falls back to the global
    empirical envelope of the trained shots."""
    rows = [
        (400.0, 700.0, 410.0, 690.0),
        (400.0, 700.0, 420.0, 700.0),
        (400.0, 700.0, 430.0, 710.0),
        (400.0, 700.0, 440.0, 720.0),
        (400.0, 700.0, 450.0, 730.0),
        (400.0, 700.0, 460.0, 740.0),
        (400.0, 700.0, 470.0, 750.0),
        (400.0, 700.0, 480.0, 760.0),
    ]
    pred = fit_trajectory_knn(rows, k=3)
    assert pred is not None
    # Even a hoop far outside the trained range can't pull the scan
    # past [410, 480].
    py_lo = pred.predict(400, 100)
    py_hi = pred.predict(400, 1000)
    assert 410.0 <= py_lo <= 480.0
    assert 410.0 <= py_hi <= 480.0


def test_trajectory_knn_envelope_clamps_scan_to_nearby_makes():
    """When makes are supplied, the scan's range for a query is bounded
    by the M nearest makes' min/max platform_y. A pathological shot set
    that would otherwise produce a scan-edge "winner" at py=300 still
    returns a value inside the nearby-makes envelope (440..460)."""
    # 200 noisy shots scattered all over platform_y. Without an envelope,
    # the scan would readily find a coincidental zero-crossing at the
    # extremes.
    rows: list[tuple[float, float, float, float]] = []
    for hoop_y in (380.0, 400.0, 420.0):
        for hoop_x in (600.0, 650.0, 700.0):
            for py in range(300, 510, 10):
                # A noisy/biased landing function: reduces toward 0 as
                # platform_y approaches 300 (mimicking the scan-edge
                # extrapolation problem from the real DB).
                landing = hoop_x + (py - 405) * 1.5
                rows.append((hoop_y, hoop_x, float(py), landing))
    # Makes cluster tightly around platform_y in [440, 460].
    makes = [
        (400.0, 650.0, 445.0),
        (400.0, 650.0, 450.0),
        (400.0, 650.0, 455.0),
        (400.0, 650.0, 460.0),
        (400.0, 650.0, 440.0),
    ]
    pred = fit_trajectory_knn(rows, makes=makes, k=5)
    assert pred is not None
    py = pred.predict(400, 650)
    # 10 px envelope margin on each side → scan in [430, 470].
    assert 430.0 <= py <= 470.0


def test_trajectory_knn_envelope_per_query():
    """Different queries see different nearest makes, so different
    envelopes. A query near makes-cluster A picks from A's range; a
    query near cluster B picks from B's range. Each cluster has at
    least envelope_k=5 makes so 5-NN stays local."""
    rows: list[tuple[float, float, float, float]] = []
    for py in range(300, 510, 10):
        rows.append((400.0, 600.0, float(py), 600.0 + (py - 405) * 1.5))
        rows.append((400.0, 700.0, float(py), 700.0 + (py - 405) * 1.5))
    makes = [
        (400.0, 600.0, 350.0), (400.0, 600.0, 355.0), (400.0, 600.0, 360.0),
        (400.0, 600.0, 365.0), (400.0, 600.0, 370.0),
        (400.0, 700.0, 470.0), (400.0, 700.0, 475.0), (400.0, 700.0, 480.0),
        (400.0, 700.0, 485.0), (400.0, 700.0, 490.0),
    ]
    pred = fit_trajectory_knn(rows, makes=makes, k=5)
    assert pred is not None
    py_low_cluster = pred.predict(400, 600)
    py_high_cluster = pred.predict(400, 700)
    # Each query stays inside its local makes envelope (350..370 for the
    # low cluster, 470..490 for the high cluster, ±10 px margin).
    assert 340.0 <= py_low_cluster <= 380.0
    assert 460.0 <= py_high_cluster <= 500.0


def test_fit_trajectory_gp_returns_none_with_too_few_samples():
    pytest.importorskip("sklearn")
    rows = [(400.0, 700.0, 410.0, 690.0)] * 5
    assert fit_trajectory_gp(rows, min_samples=12) is None


def test_trajectory_gp_recovers_optimal_platform_y():
    """Synthetic data where landing_x = hoop_x + 0.5*(py-420). The
    optimal py for landing == hoop_x is 420. The GP scan should find
    it within one step at a query point dense in training data."""
    pytest.importorskip("sklearn")
    rows = _synthetic_trajectory_rows()
    pred = fit_trajectory_gp(rows, min_samples=10, py_scan_step=5.0)
    assert pred is not None
    py_pred = pred.predict(400, 600)
    assert abs(py_pred - 420.0) <= 5.0


def test_trajectory_gp_envelope_clamps_scan_to_nearby_makes():
    """As with TrajectoryKnnPredictor: when nearby makes are concentrated
    in a tight platform_y band, the scan stays in that band even when
    the model surface would otherwise let an extreme py 'win' on
    landing match."""
    pytest.importorskip("sklearn")
    rows: list[tuple[float, float, float, float]] = []
    for hoop_y in (380.0, 400.0, 420.0):
        for hoop_x in (600.0, 650.0, 700.0):
            for py in range(300, 510, 10):
                landing = hoop_x + (py - 405) * 1.5
                rows.append((hoop_y, hoop_x, float(py), landing))
    makes = [
        (400.0, 650.0, 445.0),
        (400.0, 650.0, 450.0),
        (400.0, 650.0, 455.0),
        (400.0, 650.0, 460.0),
        (400.0, 650.0, 440.0),
    ]
    pred = fit_trajectory_gp(rows, makes=makes, min_samples=10)
    assert pred is not None
    py = pred.predict(400, 650)
    assert 430.0 <= py <= 470.0


def test_trajectory_gp_predict_landing_with_std_returns_two_floats():
    """Posterior (mean, std) at a training-near point — both finite,
    std non-negative."""
    pytest.importorskip("sklearn")
    rows = _synthetic_trajectory_rows()
    pred = fit_trajectory_gp(rows, min_samples=10)
    assert pred is not None
    mu, std = pred.predict_landing_with_std(400, 600, 420)
    assert isinstance(mu, float) and isinstance(std, float)
    assert std >= 0


def test_trajectory_gp_matches_predictor_protocol():
    """Same .n / .predict surface as the other predictors."""
    pytest.importorskip("sklearn")
    rows = _synthetic_trajectory_rows()
    pred = fit_trajectory_gp(rows, min_samples=10)
    assert pred is not None
    assert isinstance(pred.n, int)
    assert isinstance(pred.predict(400, 600), float)


def test_trajectory_knn_matches_predictor_protocol():
    """Same .n / .predict surface as the other predictors so the call
    site at minigames/hoops/main.py:52 doesn't need a special case."""
    rows = _synthetic_trajectory_rows()
    pred = fit_trajectory_knn(rows, k=3)
    assert pred is not None
    assert isinstance(pred.n, int)
    assert isinstance(pred.predict(400, 700), float)
