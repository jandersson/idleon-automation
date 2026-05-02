"""Tests for hoops _compute_offset (uses bivariate predictor)."""
from minigames.hoops.main import _compute_offset, COLD_START_OFFSET


def test_cold_start_returns_constant_offset():
    assert _compute_offset(300, 700, None) == COLD_START_OFFSET
    assert _compute_offset(450, 600, None) == COLD_START_OFFSET


def test_predictor_drives_offset():
    # target_y = 1.0*hoop_y + 0.0*hoop_x + 20 → offset = 20 everywhere
    predictor = (1.0, 0.0, 20.0, 5)
    assert _compute_offset(300, 700, predictor) == 20
    assert _compute_offset(450, 600, predictor) == 20


def test_predictor_uses_hoop_x():
    # target_y = 1.0*hoop_y - 0.1*hoop_x + 100
    # → offset = -0.1*hoop_x + 100
    predictor = (1.0, -0.1, 100.0, 5)
    # hoop_x=600 → -60 + 100 = 40
    assert _compute_offset(450, 600, predictor) == 40
    # hoop_x=700 → -70 + 100 = 30
    assert _compute_offset(450, 700, predictor) == 30
