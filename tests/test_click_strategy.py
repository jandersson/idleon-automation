"""Tests for the per-shot click position picker."""
from minigames.hoops import main as hoops_main
from minigames.hoops.main import _pick_click_position, VARIED_CLICK_POSITIONS


def test_default_hoop_strategy_clicks_at_hoop(monkeypatch):
    monkeypatch.setattr(hoops_main, "CLICK_STRATEGY", "hoop")
    assert _pick_click_position(700, 450, 960, 572, shot_idx=1) == (700, 450)
    assert _pick_click_position(617, 372, 960, 572, shot_idx=5) == (617, 372)


def test_center_strategy_returns_window_center(monkeypatch):
    monkeypatch.setattr(hoops_main, "CLICK_STRATEGY", "center")
    assert _pick_click_position(700, 450, 960, 572, shot_idx=1) == (480, 286)


def test_varied_strategy_cycles_through_positions(monkeypatch):
    monkeypatch.setattr(hoops_main, "CLICK_STRATEGY", "varied")
    hoop_x, hoop_y = 700, 450
    expected = {
        "hoop":       (700, 450),
        "center":     (480, 286),
        "above_hoop": (700, 370),
        "far_right":  (800, 286),
    }
    for i, label in enumerate(VARIED_CLICK_POSITIONS):
        got = _pick_click_position(hoop_x, hoop_y, 960, 572, shot_idx=i + 1)
        assert got == expected[label], f"shot {i+1} ({label}): {got!r} != {expected[label]!r}"


def test_above_hoop_clamps_to_zero(monkeypatch):
    monkeypatch.setattr(hoops_main, "CLICK_STRATEGY", "varied")
    # hoop_y=20, above_hoop wants hoop_y - 80 = -60, should clamp to 0.
    # above_hoop is index 2 in the cycle, so shot_idx=3.
    got = _pick_click_position(700, 20, 960, 572, shot_idx=3)
    assert got == (700, 0)
