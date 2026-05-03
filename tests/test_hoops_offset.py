"""Tests for hoops _compute_offset (uses a Predictor from common.predictor),
the miss-driven perturbation sweep, and the make-detection helper."""
import numpy as np

from common.predictor import KnnPredictor
from minigames.hoops.main import (
    _compute_offset, COLD_START_OFFSET, _perturbation_for, PERTURBATION_SEQUENCE,
    _log_shot_result,
)


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


def _fake_score_crop(value: int) -> np.ndarray:
    """Tiny arbitrary grayscale crop. score_changed needs std >= 5 in
    EITHER pre or post; we make it noisy enough to pass that check.
    The actual content doesn't matter — _log_shot_result's decision
    comes from score_after_int / score_before_int, not the crop diff."""
    rng = np.random.RandomState(value).randint(0, 256, (10, 20), dtype=np.uint8)
    return rng


def test_log_shot_reanchors_session_score_from_pre_ocr():
    """If the user manually scored mid-session, the bot's session_score
    is stale. The next shot's pre-OCR captures the new real score; we
    re-anchor on that so the bot doesn't credit itself for the
    user's points."""
    stats = {"makes": 0, "attempts": 0, "session_score": 0}
    pre = _fake_score_crop(1)
    post = _fake_score_crop(2)
    # User scored 1 manually before this shot. Pre OCR sees 1.
    # This shot also misses (sa stays 1). Without re-anchoring, the
    # bot would compute increment = 1 - 0 = 1 → false MAKE credit.
    # With the fix, session_score advances to 1 first → increment = 0.
    made, _, inc = _log_shot_result(
        stats, pre, post,
        score_before_int=1, score_after_int=1,
        ball_x_at_rim=None, hoop_x=None,
    )
    assert made is False
    assert inc == 0
    assert stats["session_score"] == 1  # anchor advanced to match reality
    assert stats["makes"] == 0


def test_log_shot_normal_make_after_reanchor():
    """Normal make math still works after the anchor catches up."""
    stats = {"makes": 0, "attempts": 0, "session_score": 1}
    pre = _fake_score_crop(1)
    post = _fake_score_crop(2)
    made, _, inc = _log_shot_result(
        stats, pre, post,
        score_before_int=1, score_after_int=2,
        ball_x_at_rim=None, hoop_x=None,
    )
    assert made is True
    assert inc == 1
    assert stats["session_score"] == 2
    assert stats["makes"] == 1
