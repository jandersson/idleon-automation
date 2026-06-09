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


def test_perturbation_first_shot_returns_zero_regardless_of_sigma():
    # miss_count=0 means it's the predictor's first shot at this hoop —
    # never perturb, regardless of how uncertain the predictor is.
    for sigma in (None, 0.0, 15.0, 35.0, 100.0):
        assert _perturbation_for(0, sigma) == 0


def test_perturbation_low_sigma_matches_unscaled_sequence():
    # σ below the first bucket means scale=1.0 → identical to no-σ path.
    for i in range(len(PERTURBATION_SEQUENCE)):
        assert _perturbation_for(i, sigma=None) == _perturbation_for(i, sigma=5.0)


def test_perturbation_high_sigma_scales_magnitudes_up():
    # σ in the 20-40 bucket → scale 2.0; σ >=40 → scale 3.0. Higher σ
    # means larger initial steps so the bot walks to the make zone
    # faster when the predictor isn't confident.
    assert _perturbation_for(1, sigma=25.0) == -16  # -8 * 2
    assert _perturbation_for(1, sigma=50.0) == -24  # -8 * 3
    assert _perturbation_for(2, sigma=50.0) == 24   # +8 * 3
    assert _perturbation_for(3, sigma=50.0) == -32  # -16 * 3 = -48, capped


def test_perturbation_capped_at_max():
    from minigames.hoops.main import PERTURBATION_MAX
    for i in range(len(PERTURBATION_SEQUENCE) + 5):
        for sigma in (None, 15.0, 25.0, 50.0):
            assert abs(_perturbation_for(i, sigma)) <= PERTURBATION_MAX


def test_sweep_advances_on_way_off_miss():
    from minigames.hoops.main import _sweep_should_advance
    assert _sweep_should_advance(500, 700)   # 200px short
    assert _sweep_should_advance(900, 700)   # 200px long


def test_sweep_holds_on_in_band_miss():
    # Ball arrived within IN_BAND_RESIDUAL_PX of the hoop — aim was right,
    # the miss was rim luck. Re-fire the same target.
    from minigames.hoops.main import _sweep_should_advance, IN_BAND_RESIDUAL_PX
    assert not _sweep_should_advance(700, 700)
    assert not _sweep_should_advance(700 - IN_BAND_RESIDUAL_PX, 700)
    assert not _sweep_should_advance(700 + IN_BAND_RESIDUAL_PX, 700)
    assert _sweep_should_advance(700 + IN_BAND_RESIDUAL_PX + 1, 700)


def test_sweep_advances_without_trajectory_data():
    # No measurement → can't distinguish mislaunch from near-miss; advance
    # rather than risk a stuck loop on an unmeasured target.
    from minigames.hoops.main import _sweep_should_advance
    assert _sweep_should_advance(None, 700)
    assert _sweep_should_advance(650, None)


def test_perturbation_sign_preserved_across_scales():
    # Scaling magnitudes must not flip the alternating ± pattern that
    # walks the sweep outward in both directions.
    for sigma in (None, 5.0, 15.0, 30.0, 100.0):
        signs = [
            (1 if _perturbation_for(i, sigma) > 0 else (-1 if _perturbation_for(i, sigma) < 0 else 0))
            for i in range(1, 9)
        ]
        # Same sign pattern as the unscaled sequence (skipping the leading 0).
        base_signs = [(1 if p > 0 else -1) for p in PERTURBATION_SEQUENCE[1:9]]
        assert signs == base_signs


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


def test_log_shot_rejects_make_when_ball_overshoots_trajectory_check():
    """OCR says score went up by 1 but the ball flew well past the hoop.
    Without prompt confirmation, the trajectory check rejects it as a
    likely OCR misread / rattle-in / bad detection."""
    from minigames.hoops.main import TRAJECTORY_MAKE_TOLERANCE
    stats = {"makes": 0, "attempts": 0, "session_score": 0}
    pre = _fake_score_crop(0)
    post = _fake_score_crop(1)
    made, _, _ = _log_shot_result(
        stats, pre, post,
        score_before_int=0, score_after_int=1,
        ball_x_at_rim=700 + TRAJECTORY_MAKE_TOLERANCE + 10, hoop_x=700,
    )
    assert made is False
    assert stats["makes"] == 0


def test_log_shot_prompt_disappeared_overrides_trajectory_rejection():
    """If the in-game 'Make a shot to start' prompt cleared after a shot,
    that's binary truth — count it as a make even if the ball landed too
    far from the hoop for the trajectory check to like it (e.g. bank
    shots that drop through 70-80 px off-center)."""
    from minigames.hoops.main import TRAJECTORY_MAKE_TOLERANCE
    stats = {"makes": 0, "attempts": 0, "session_score": 0}
    pre = _fake_score_crop(0)
    post = _fake_score_crop(1)
    made, _, _ = _log_shot_result(
        stats, pre, post,
        score_before_int=None, score_after_int=1,
        ball_x_at_rim=700 + TRAJECTORY_MAKE_TOLERANCE + 10, hoop_x=700,
        prompt_disappeared=True,
    )
    assert made is True
    assert stats["makes"] == 1
    assert stats["session_score"] == 1


def test_estimate_bob_period_returns_none_for_short_buffer():
    from minigames.hoops.main import _estimate_bob_period_ms
    assert _estimate_bob_period_ms([(0.0, 136, 400)]) is None


def test_estimate_bob_period_recovers_known_period_from_synthetic_sine():
    """Build a 200ms-period sinusoid sampled at 10ms intervals; the
    detector should return ~200ms ± a small margin."""
    import math
    from minigames.hoops.main import _estimate_bob_period_ms
    samples = []
    for i in range(60):
        t = i * 0.010
        py = int(400 + 80 * math.sin(2 * math.pi * t / 0.200))
        samples.append((t, 136, py))
    period_ms = _estimate_bob_period_ms(samples)
    assert period_ms is not None
    assert 180 <= period_ms <= 220, f"got {period_ms}"


def test_estimate_bob_period_returns_none_for_flat_signal():
    """A platform stuck at one y (e.g. clamped at the bob extreme) has
    no peaks and no period."""
    from minigames.hoops.main import _estimate_bob_period_ms
    samples = [(i * 0.01, 136, 510) for i in range(60)]
    assert _estimate_bob_period_ms(samples) is None


def test_estimate_bob_period_filters_jitter_below_prominence():
    """A signal with only ±2px jitter (no real bob) must not produce a
    spurious period — min_prominence_px=8 default cuts those."""
    import random
    from minigames.hoops.main import _estimate_bob_period_ms
    rng = random.Random(0)
    samples = [(i * 0.01, 136, 400 + rng.randint(-2, 2)) for i in range(60)]
    assert _estimate_bob_period_ms(samples) is None


def test_log_shot_prompt_alone_does_not_create_make_without_score_increment():
    """The prompt-disappearance signal is only used to override the
    trajectory check. If OCR shows no score change at all, we still
    classify as miss — prompt detection is template-matched and can
    flicker; the score-increment requirement keeps a single false
    negative on the prompt match from inventing a make."""
    stats = {"makes": 0, "attempts": 0, "session_score": 0}
    pre = _fake_score_crop(0)
    post = _fake_score_crop(0)
    made, _, _ = _log_shot_result(
        stats, pre, post,
        score_before_int=0, score_after_int=0,
        ball_x_at_rim=700, hoop_x=700,
        prompt_disappeared=True,
    )
    assert made is False
    assert stats["makes"] == 0


def test_platform_velocity_recovers_constant_slope():
    """Platform descending at 50 px/s, sampled every 20ms — the
    least-squares slope must recover the velocity closely."""
    from minigames.hoops.main import _platform_velocity
    samples = [(i * 0.02, 136, int(round(400 + 50 * i * 0.02))) for i in range(20)]
    vy = _platform_velocity(samples)
    assert vy is not None
    assert abs(vy - 50.0) < 3.0


def test_platform_velocity_sign_convention():
    """Platform moving up (py decreasing) must give negative vy."""
    from minigames.hoops.main import _platform_velocity
    samples = [(i * 0.02, 136, 400 - i) for i in range(15)]  # -50 px/s
    vy = _platform_velocity(samples)
    assert vy is not None and vy < 0


def test_platform_velocity_uses_only_recent_window():
    """Old samples outside the window must not contaminate the slope:
    platform fell for a while, then held still — vy at the end is ~0."""
    from minigames.hoops.main import _platform_velocity
    falling = [(i * 0.02, 136, 400 + i * 2) for i in range(50)]   # 0..0.98s
    still = [(1.0 + i * 0.02, 136, 498) for i in range(20)]       # 1.0..1.38s
    vy = _platform_velocity(falling + still, window_s=0.3)
    assert vy is not None
    assert abs(vy) < 2.0


def test_platform_velocity_works_at_real_loop_cadence():
    """The main loop samples every ~200-1000ms (observed 2026-06-09), not
    every 20ms. A handful of sparse samples must still produce a slope —
    this is the regression test for the all-NULL platform_vy session."""
    from minigames.hoops.main import _platform_velocity
    # 50 px/s descent sampled every 400ms — 4 samples in the 1.5s window.
    samples = [(i * 0.4, 136, int(round(400 + 50 * i * 0.4))) for i in range(4)]
    vy = _platform_velocity(samples)
    assert vy is not None
    assert abs(vy - 50.0) < 3.0
    # Even two samples qualify when they span enough time.
    two = [(0.0, 136, 400), (0.5, 136, 425)]
    vy2 = _platform_velocity(two)
    assert vy2 is not None
    assert abs(vy2 - 50.0) < 3.0


def test_platform_velocity_too_few_samples_returns_none():
    from minigames.hoops.main import _platform_velocity
    assert _platform_velocity([]) is None
    assert _platform_velocity([(0.0, 136, 400)]) is None
    # Two samples 20ms apart: span below min_span_s — jitter-dominated.
    assert _platform_velocity([(0.0, 136, 400), (0.02, 136, 401)]) is None


def test_platform_velocity_tolerates_detector_jitter():
    """±2px jitter on a 50 px/s ramp shouldn't swing the estimate far."""
    import random
    from minigames.hoops.main import _platform_velocity
    rng = random.Random(0)
    samples = [
        (i * 0.02, 136, int(round(400 + 50 * i * 0.02)) + rng.randint(-2, 2))
        for i in range(16)
    ]
    vy = _platform_velocity(samples)
    assert vy is not None
    assert abs(vy - 50.0) < 15.0


def test_log_shot_prompt_visible_ignores_poisoned_pre_read():
    """Regression for session 2026-06-09 20:13: on a fresh game (start
    prompt up, true score 0) the pre-shot OCR misread 7, the anchor
    re-anchored to 7, and every later real score of 1 looked like
    increment -6 — no make could be detected all game. When the prompt
    is visible the score is definitionally 0: the anchor must reset and
    the pre-read must be ignored."""
    stats = {"makes": 0, "attempts": 0, "session_score": 0}
    # Shot 1: fresh game, poisoned pre-read, post OCR dropped out.
    made, _, _ = _log_shot_result(
        stats, _fake_score_crop(0), _fake_score_crop(1),
        score_before_int=7, score_after_int=None,
        prompt_visible_before=True,
    )
    assert made is None or made is False
    assert stats["session_score"] == 0  # anchor not poisoned
    # Shot 2: real score is now 1 — must register as a make.
    made, _, inc = _log_shot_result(
        stats, _fake_score_crop(1), _fake_score_crop(2),
        score_before_int=1, score_after_int=2,
        ball_x_at_rim=700, hoop_x=700,
    )
    assert made is True


def test_log_shot_re_anchor_still_works_without_prompt():
    """Mid-game (no prompt) the upward re-anchor from pre-OCR must keep
    working — it covers the user scoring manually between bot shots."""
    stats = {"makes": 0, "attempts": 0, "session_score": 0}
    made, _, _ = _log_shot_result(
        stats, _fake_score_crop(3), _fake_score_crop(3),
        score_before_int=3, score_after_int=3,
        prompt_visible_before=False,
    )
    assert stats["session_score"] == 3
    assert made is False


def test_clean_make_rejects_prompt_confirmed_pole_ricochet():
    """Regression for session 2026-06-09 20:13 shot 1: a first-shot
    pole-tip bounce-in (peak_x 252px past landing) is a real MAKE but
    must not become predictor training data — the old prompt-confirmed
    exemption set clean_make=1 unconditionally."""
    from minigames.hoops.main import _classify_clean_make
    clean, reason = _classify_clean_make(True, landing=635, peak_x=887, hoop_x=715)
    assert clean == 0
    assert "drift" in reason


def test_clean_make_accepts_swish():
    from minigames.hoops.main import _classify_clean_make
    clean, reason = _classify_clean_make(True, landing=720, peak_x=730, hoop_x=715)
    assert clean == 1
    assert reason is None


def test_clean_make_accepts_bank_shot_with_modest_drift():
    """Bank shots land off-center but within tolerance, with drift under
    the back-drift limit — they pass on their own merits, no prompt
    exemption needed."""
    from minigames.hoops.main import _classify_clean_make, MAX_BACK_DRIFT_PX
    clean, reason = _classify_clean_make(
        True, landing=650, peak_x=650 + MAX_BACK_DRIFT_PX, hoop_x=715,
    )
    assert clean == 1


def test_clean_make_rejects_far_landing():
    from minigames.hoops.main import _classify_clean_make
    clean, reason = _classify_clean_make(True, landing=900, peak_x=None, hoop_x=715)
    assert clean == 0
    assert "rattle-in" in reason


def test_clean_make_trusts_make_without_trajectory():
    from minigames.hoops.main import _classify_clean_make
    assert _classify_clean_make(True, landing=None, peak_x=None, hoop_x=715) == (1, None)


def test_clean_make_miss_and_unknown():
    from minigames.hoops.main import _classify_clean_make
    assert _classify_clean_make(False, landing=720, peak_x=720, hoop_x=715) == (0, None)
    assert _classify_clean_make(None, landing=720, peak_x=720, hoop_x=715) == (None, None)
