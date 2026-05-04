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
