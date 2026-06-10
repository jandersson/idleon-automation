"""Tests for common.ball_trajectory.summarise_trajectory.

The math reduction is the testable part — the I/O (loading PNGs and running
HSV detection) is integration work, easier to verify by running the bot
once than to mock here.
"""
from common.ball_trajectory import summarise_trajectory


def test_empty_trajectory_returns_all_none():
    out = summarise_trajectory([], hoop_x=700, hoop_y=450)
    assert out == {
        "ball_apex_y": None,
        "ball_peak_x": None,
        "ball_x_at_rim_height": None,
        "ball_landing_x": None,
        "ball_flight_ms": None,
    }


def test_peak_x_is_max_x():
    """Rightmost x along the whole trajectory."""
    positions = [(1, 100, 400), (2, 600, 200), (3, 300, 350)]
    out = summarise_trajectory(positions, hoop_x=400, hoop_y=450)
    assert out["ball_peak_x"] == 600


def test_peak_x_minus_landing_x_captures_backboard_bounce():
    """Ball flies past hoop, peaks at backboard post, deflects back and
    lands near hoop. peak_x - landing_x is the bounce signature."""
    hoop_x, hoop_y = 589, 419
    positions = [
        (1, 200, 400),
        (2, 500, 200),  # ascending
        (3, 700, 350),  # past hoop, still ascending in x
        (4, 887, 400),  # peak — slammed into backboard post
        (5, 750, 430),  # bouncing back
        (6, 603, 500),  # landing in hoop, well left of peak
    ]
    out = summarise_trajectory(positions, hoop_x, hoop_y)
    assert out["ball_peak_x"] == 887
    assert out["ball_landing_x"] == 603
    assert out["ball_peak_x"] - out["ball_landing_x"] == 284


def test_ball_flight_ms_spans_first_to_last_frame():
    # Frame indices 1 → 9 = 8 frame intervals at the default 50ms.
    positions = [(1, 100, 400), (3, 200, 200), (9, 300, 350)]
    out = summarise_trajectory(positions, hoop_x=400, hoop_y=450)
    assert out["ball_flight_ms"] == 8 * 50


def test_apex_is_min_y():
    # y=200 is "highest" on screen (smallest y).
    positions = [(1, 100, 400), (2, 200, 200), (3, 300, 350)]
    out = summarise_trajectory(positions, hoop_x=400, hoop_y=450)
    assert out["ball_apex_y"] == 200


def test_landing_x_is_last_x():
    positions = [(1, 100, 400), (2, 200, 200), (3, 300, 350)]
    out = summarise_trajectory(positions, hoop_x=400, hoop_y=450)
    assert out["ball_landing_x"] == 300


def test_x_at_rim_height_picks_position_closest_to_hoop_y():
    hoop_y = 450
    # Position with y=445 is 5px from hoop_y, well within tolerance.
    positions = [(1, 100, 200), (2, 250, 445), (3, 400, 600)]
    out = summarise_trajectory(positions, hoop_x=300, hoop_y=hoop_y)
    assert out["ball_x_at_rim_height"] == 250


def test_x_at_rim_height_none_when_no_position_close_enough():
    # All positions are far from hoop_y=450.
    positions = [(1, 100, 200), (2, 250, 700), (3, 400, 100)]
    out = summarise_trajectory(positions, hoop_x=300, hoop_y=450)
    assert out["ball_x_at_rim_height"] is None


def test_undershoot_pattern():
    """Ball arcs up, comes down 50px short of hoop_x at rim height."""
    hoop_x, hoop_y = 700, 450
    positions = [
        (1, 200, 400),  # leaves player
        (2, 350, 200),  # apex
        (3, 500, 350),  # descending
        (4, 650, 445),  # ← at rim height, 50px short
        (5, 700, 540),  # below rim, past front
    ]
    out = summarise_trajectory(positions, hoop_x, hoop_y)
    assert out["ball_apex_y"] == 200
    assert out["ball_x_at_rim_height"] == 650  # 50px short of hoop_x
    assert out["ball_landing_x"] == 700


def test_overshoot_pattern():
    """Ball goes past the rim horizontally."""
    hoop_x, hoop_y = 700, 450
    positions = [
        (1, 200, 400),
        (2, 400, 200),
        (3, 600, 350),
        (4, 750, 450),  # ← at rim height, 50px past
        (5, 800, 540),
    ]
    out = summarise_trajectory(positions, hoop_x, hoop_y)
    assert out["ball_x_at_rim_height"] == 750


def test_ui_artifact_corner_is_excluded():
    """The 'Current Reward' counter (top-right) animates on makes and was
    detected as the ball, stamping peak_x ~887 and faking backward drift
    that disqualified ~20% of training makes (found 2026-06-10)."""
    from common.ball_trajectory import _is_ui_artifact
    w, h = 960, 572
    assert _is_ui_artifact(887, 50, w, h)        # the reward counter
    assert _is_ui_artifact(940, 80, w, h)
    assert not _is_ui_artifact(887, 300, w, h)   # descending overshoot ball
    assert not _is_ui_artifact(600, 40, w, h)    # real apex mid-court
    assert not _is_ui_artifact(700, 50, w, h)


def test_track_drops_ui_corner_detections(monkeypatch):
    """A detection inside the excluded corner never enters the trajectory."""
    import numpy as np
    import common.ball_trajectory as bt
    fake_balls = [(887, 50), (600, 200), (650, 250)]
    calls = iter(fake_balls)
    monkeypatch.setattr(bt, "find_ball",
                        lambda *a, **k: next(calls, None))
    frames = [np.zeros((572, 960, 3), dtype=np.uint8) for _ in range(4)]
    positions = bt.track_ball_trajectory(frames)
    xs = [p[1] for p in positions]
    assert 887 not in xs
    assert xs == [600, 650]
