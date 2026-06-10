"""Tests for minigames.darts.detector — game-over template fallback path."""
from pathlib import Path

import numpy as np
import pytest

from minigames.darts import detector


def test_find_game_over_returns_false_when_template_missing(tmp_path, monkeypatch):
    """If assets/game_over.png hasn't been captured yet, the function
    must early-return (False, 0.0) so the main loop can fall through to
    the no-pose timeout heuristic instead of crashing on a missing
    template."""
    monkeypatch.setattr(detector, "ASSETS", tmp_path)
    frame = np.zeros((100, 200, 4), dtype=np.uint8)  # BGRA
    is_over, conf = detector.find_game_over(frame)
    assert is_over is False
    assert conf == 0.0


def test_find_game_over_returns_false_when_template_unreadable(tmp_path, monkeypatch):
    """A zero-byte / corrupt template file shouldn't crash either —
    cv2.imread returns None for unreadable files, and the function
    should treat that the same as 'template missing'."""
    monkeypatch.setattr(detector, "ASSETS", tmp_path)
    (tmp_path / "game_over.png").write_bytes(b"")  # not a valid image
    frame = np.zeros((100, 200, 4), dtype=np.uint8)
    is_over, conf = detector.find_game_over(frame)
    assert is_over is False
    assert conf == 0.0


def _frame_with_dart(dart_y: int, dart_x: int = 200, size=(400, 400)) -> np.ndarray:
    """Noisy background with a distinctive textured 'dart' patch at (x, y).
    The patch must have internal variance — TM_CCOEFF_NORMED is degenerate
    on constant templates."""
    rng = np.random.RandomState(7)
    img = rng.randint(0, 60, (*size, 3), dtype=np.uint8)
    patt = ((np.indices((8, 24)).sum(axis=0) * 31) % 180 + 70).astype(np.uint8)
    img[dart_y - 4:dart_y + 4, dart_x - 12:dart_x + 12] = np.stack([patt] * 3, axis=-1)
    return img


def test_compute_dart_dy_measures_vertical_motion():
    """Dart moved down 9px between prev and cur frame: dy = +9.
    Regression scaffolding for the #26 swing-pass discriminator."""
    from minigames.darts.detector import compute_dart_dy
    prev = _frame_with_dart(191)
    cur = _frame_with_dart(200)
    template = cur[196:204, 188:212].copy()  # crop the dart from cur
    dy = compute_dart_dy(prev, 200, 200, template=template)
    assert dy is not None
    assert abs(dy - 9) <= 1


def test_compute_dart_dy_upswing_is_negative():
    from minigames.darts.detector import compute_dart_dy
    prev = _frame_with_dart(230)
    cur = _frame_with_dart(200)  # dart moved UP 30px
    template = cur[196:204, 188:212].copy()
    dy = compute_dart_dy(prev, 200, 200, template=template)
    assert dy is not None
    assert abs(dy - (-30)) <= 1


def test_compute_dart_dy_none_without_prev_or_match():
    from minigames.darts.detector import compute_dart_dy
    cur = _frame_with_dart(200)
    template = cur[196:204, 188:212].copy()
    assert compute_dart_dy(None, 200, 200, template=template) is None
    # Prev frame has no dart anywhere near the window → no confident match.
    rng = np.random.RandomState(8)
    empty = rng.randint(0, 60, (400, 400, 3), dtype=np.uint8)
    assert compute_dart_dy(empty, 200, 200, template=template) is None


def test_match_or_save_wind_sample_returns_stable_name(tmp_path, monkeypatch):
    """Throws must record WHICH wind state they flew under — the
    wind_sample column was silently NULL on every throw because the old
    helper only returned a saved/not-saved bool (#26 follow-up work needs
    the association)."""
    import minigames.darts.main as dm
    monkeypatch.setattr(dm, "WIND_SAMPLES_DIR", tmp_path)
    rng = np.random.RandomState(3)
    state_a = rng.randint(0, 255, (20, 60, 3), dtype=np.uint8)
    state_b = rng.randint(0, 255, (20, 60, 3), dtype=np.uint8)
    seen = []
    name_a = dm._match_or_save_wind_sample(state_a, seen)
    assert name_a is not None and (tmp_path / name_a).exists()
    # Same state again (tiny noise below dedup threshold) → same name.
    jitter = state_a.astype(np.int16) + rng.randint(-2, 3, state_a.shape)
    assert dm._match_or_save_wind_sample(
        jitter.clip(0, 255).astype(np.uint8), seen) == name_a
    # A different state gets its own name.
    name_b = dm._match_or_save_wind_sample(state_b, seen)
    assert name_b is not None and name_b != name_a
    # No crop (wind region not picked) → None.
    assert dm._match_or_save_wind_sample(None, seen) is None


def test_dy_gate_skips_upswing_fires_only():
    """The #26 fire gate: positive centroid-dy = up-swing pass (skip);
    negative or zero fires; None fires (instrument dropout must not
    starve the bot). Thresholds from the validated record: up-swing
    misses at dy +6..+12, release hits at dy -7..-13."""
    from minigames.darts.main import _is_upswing_fire, DY_GATE_MAX
    assert _is_upswing_fire(7)            # tonight's miss signature
    assert _is_upswing_fire(DY_GATE_MAX + 1)
    assert not _is_upswing_fire(-8)       # release-pass hit signature
    assert not _is_upswing_fire(DY_GATE_MAX)
    assert not _is_upswing_fire(None)     # dropout → fire, don't starve


def test_find_celebration_on_real_frames():
    """The streak banner ('...one hundred and EIGHTY!', 4 bullseyes in a
    row) replaced the throwing pose and made the no-pose timeout bail
    mid-game (session 2026-06-10 11:52). Template-matched against the
    captured frame; normal-play frames stay under threshold."""
    from pathlib import Path
    import cv2
    from minigames.darts.detector import find_celebration
    root = Path(__file__).parent.parent
    pos = root / "minigames/darts/assets/monitor/throw_003_115303/post_throw.png"
    neg = root / "minigames/darts/assets/monitor/throw_002_115254/post_throw.png"
    if not (pos.exists() and neg.exists()):
        return  # monitor frames cleaned up — skip
    pos_bgra = cv2.cvtColor(cv2.imread(str(pos)), cv2.COLOR_BGR2BGRA)
    neg_bgra = cv2.cvtColor(cv2.imread(str(neg)), cv2.COLOR_BGR2BGRA)
    is_celeb, conf = find_celebration(pos_bgra)
    assert is_celeb and conf > 0.9
    is_celeb_neg, conf_neg = find_celebration(neg_bgra)
    assert not is_celeb_neg and conf_neg < 0.6
