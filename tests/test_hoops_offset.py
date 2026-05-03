"""Tests for hoops _compute_offset (uses a Predictor from common.predictor)
and the miss-driven perturbation sweep."""
from common.predictor import KnnPredictor
from minigames.hoops.main import _compute_offset, COLD_START_OFFSET, _perturbation_for, PERTURBATION_SEQUENCE


def test_cold_start_returns_constant_offset():
    assert _compute_offset(300, 700, None) == COLD_START_OFFSET
    assert _compute_offset(450, 700, None) == COLD_START_OFFSET


def test_predictor_drives_offset():
    # Past makes all at platform_y = hoop_y + 20. With k=1, the nearest
    # neighbour wins outright and we get offset=20 exactly.
    points = [
        (400.0, 700.0, 420.0),
        (410.0, 700.0, 430.0),
        (390.0, 700.0, 410.0),
        (400.0, 710.0, 420.0),
    ]
    predictor = KnnPredictor(points, k=1)
    assert _compute_offset(400, 700, predictor) == 20


def test_predictor_uses_hoop_x():
    # Two clusters: low hoop_x → small offset, high hoop_x → bigger offset.
    points = [
        (400.0, 600.0, 400.0),  # offset 0
        (400.0, 800.0, 440.0),  # offset 40
    ]
    predictor = KnnPredictor(points, k=1)  # k=1 — pure nearest
    assert _compute_offset(400, 600, predictor) == 0
    assert _compute_offset(400, 800, predictor) == 40
    # Equidistant query: returns the average (both weight 1.0/dist=1.0/100).
    # at (400, 700): nearest with k=1 is one of the two (tie-broken by sort).
    # With k=2 (override), result is the average → predicted py ~420 → offset 20.
    predictor_k2 = KnnPredictor(points, k=2)
    assert _compute_offset(400, 700, predictor_k2) == 20


def test_perturbation_zero_on_first_attempt():
    assert _perturbation_for(0) == 0


def test_perturbation_sweeps_outward():
    # First miss tries one offset, second tries the opposite, then larger.
    seq = [_perturbation_for(i) for i in range(len(PERTURBATION_SEQUENCE))]
    assert seq == PERTURBATION_SEQUENCE
    # Magnitudes are non-decreasing.
    mags = [abs(p) for p in seq]
    assert all(a <= b for a, b in zip(mags, mags[1:]))


def test_perturbation_clamps_at_end_of_sequence():
    # Past the end of the sequence: stay at the last (largest) value, don't crash.
    big_miss_count = len(PERTURBATION_SEQUENCE) + 5
    assert _perturbation_for(big_miss_count) == PERTURBATION_SEQUENCE[-1]
