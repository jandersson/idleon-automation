"""Tests for hoops _compute_offset (now uses a fitted predictor)."""
from minigames.hoops.main import _compute_offset, COLD_START_OFFSET


def test_cold_start_returns_constant_offset():
    """No predictor → fall back to the cold-start constant for any hoop_y."""
    assert _compute_offset(300, None) == COLD_START_OFFSET
    assert _compute_offset(450, None) == COLD_START_OFFSET
    assert _compute_offset(700, None) == COLD_START_OFFSET


def test_predictor_drives_offset():
    # target_y = 1.0 * hoop_y + 20 → offset = 20 everywhere
    predictor = (1.0, 20.0, 5)
    assert _compute_offset(300, predictor) == 20
    assert _compute_offset(450, predictor) == 20
    assert _compute_offset(700, predictor) == 20


def test_predictor_with_nontrivial_slope():
    # target_y = 0.9*hoop_y + 60 → offset = -0.1*hoop_y + 60
    predictor = (0.9, 60.0, 10)
    assert _compute_offset(300, predictor) == 30  # -0.1*300 + 60 = 30
    assert _compute_offset(500, predictor) == 10  # -0.1*500 + 60 = 10
