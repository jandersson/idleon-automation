"""Tests for common.predictor — the swappable Predictor implementations."""
from common.predictor import (
    KnnPredictor, BivariatePredictor, fit_knn, fit_bivariate,
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
