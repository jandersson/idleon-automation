"""Tests for hoops _compute_offset (uses a fitted platform_y predictor +
sampling-rate compensation)."""
from minigames.hoops.main import _compute_offset, COLD_START_OFFSET, UPSTROKE_COMPENSATION


def test_cold_start_returns_constant_offset():
    assert _compute_offset(300, None) == COLD_START_OFFSET
    assert _compute_offset(450, None) == COLD_START_OFFSET
    assert _compute_offset(700, None) == COLD_START_OFFSET


def test_predictor_drives_offset_with_compensation():
    # optimal_platform_y = hoop_y + 20 → target_y = hoop_y + 20 + COMPENSATION
    # → offset = 20 + COMPENSATION everywhere
    predictor = (1.0, 20.0, 5)
    expected = 20 + UPSTROKE_COMPENSATION
    assert _compute_offset(300, predictor) == expected
    assert _compute_offset(450, predictor) == expected
    assert _compute_offset(700, predictor) == expected


def test_predictor_with_nontrivial_slope():
    # optimal_platform_y = 0.9*hoop_y + 60 → target_y = 0.9*hoop_y + 60 + COMP
    # → offset = -0.1*hoop_y + 60 + COMP
    predictor = (0.9, 60.0, 10)
    assert _compute_offset(300, predictor) == 30 + UPSTROKE_COMPENSATION  # -30 + 60
    assert _compute_offset(500, predictor) == 10 + UPSTROKE_COMPENSATION  # -50 + 60
