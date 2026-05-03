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
        "ball_x_at_rim_height": None,
        "ball_landing_x": None,
    }


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
