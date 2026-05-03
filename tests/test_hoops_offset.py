"""Tests for hoops _compute_offset (uses bivariate predictor) and the
miss-driven perturbation sweep."""
from minigames.hoops.main import _compute_offset, COLD_START_OFFSET, _perturbation_for, PERTURBATION_SEQUENCE


def test_cold_start_returns_constant_offset():
    # Use hoops outside the stuck-region override.
    assert _compute_offset(300, 700, None) == COLD_START_OFFSET
    assert _compute_offset(450, 700, None) == COLD_START_OFFSET


def test_predictor_drives_offset():
    # target_y = 1.0*hoop_y + 0.0*hoop_x + 20 → offset = 20 everywhere
    predictor = (1.0, 0.0, 20.0, 5)
    assert _compute_offset(300, 700, predictor) == 20
    assert _compute_offset(450, 700, predictor) == 20


def test_predictor_uses_hoop_x():
    # target_y = 1.0*hoop_y - 0.1*hoop_x + 100
    # → offset = -0.1*hoop_x + 100. Use hoop_x in 700+ range to avoid
    # the stuck-region override (hoop_x<620, hoop_y>380).
    predictor = (1.0, -0.1, 100.0, 5)
    # hoop_x=700 → -70 + 100 = 30
    assert _compute_offset(450, 700, predictor) == 30
    # hoop_x=800 → -80 + 100 = 20
    assert _compute_offset(450, 800, predictor) == 20


def test_stuck_region_overrides():
    """Hardcoded overrides for hoop_x<660 zones where the bivariate fit
    is biased. Predictor below would otherwise return 0 for these inputs."""
    predictor = (1.0, 0.0, 0.0, 5)
    # hoop_x<620, hoop_y>380 → 95
    assert _compute_offset(410, 586, predictor) == 95
    assert _compute_offset(400, 600, predictor) == 95
    assert _compute_offset(450, 615, None) == 95
    # hoop_x<620, hoop_y<=380 → 0 (matches predictor here, but the rule
    # short-circuits regardless of predictor output)
    assert _compute_offset(330, 586, predictor) == 0
    assert _compute_offset(380, 600, None) == 0  # override beats cold-start
    # 620<=hoop_x<660, hoop_y<=380 → 5
    assert _compute_offset(330, 625, predictor) == 5
    assert _compute_offset(370, 655, None) == 5
    # Just outside the boxes — predictor takes over.
    assert _compute_offset(410, 660, predictor) == 0  # hoop_x on boundary
    assert _compute_offset(381, 600, predictor) == 0  # hoop_y on the high boundary


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
