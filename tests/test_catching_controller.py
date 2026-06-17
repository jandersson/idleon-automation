"""Tests for the catching predictive flap timer (controller.py, #60).

Two halves: the pure physics/decision functions against closed-form values,
and the dynamics fitter against a synthetic trace generated from KNOWN
dynamics (round-trip recovery). No live game; the controller is validated on
real traces by the user.
"""
import math

import pytest

from minigames.catching.controller import (
    Dynamics,
    step,
    descent_time_to,
    predict_descent_window,
    crossing_window,
    should_flap_now,
    fit_gravity,
    fit_flap_vy,
    fit_approach_speed,
    fit_dynamics,
)


DYN = Dynamics(gravity=600.0, flap_vy=-300.0, approach_speed=200.0)


# --- physics --------------------------------------------------------------

def test_step_matches_closed_form():
    y, vy = step(100.0, -50.0, 0.1, DYN)
    # y = 100 + (-50)(0.1) + 0.5*600*0.01 = 100 - 5 + 3 = 98
    # vy = -50 + 600*0.1 = 10
    assert y == pytest.approx(98.0)
    assert vy == pytest.approx(10.0)


def test_dynamics_derived_quantities():
    # rise = flap_vy^2/(2g) = 90000/1200 = 75; apex time = 300/600 = 0.5
    assert DYN.rise_height_px == pytest.approx(75.0)
    assert DYN.time_to_apex_s == pytest.approx(0.5)


def test_descent_time_to_closed_form():
    # from y0=100, flap vy=-300, g=600, reach y=85 on descent:
    # 300 t^2 - 300 t + 15 = 0 -> t = (300 + sqrt(72000))/600
    t = descent_time_to(100.0, 85.0, DYN)
    expected = (300 + math.sqrt(72000)) / 600
    assert t == pytest.approx(expected, rel=1e-6)


def test_descent_time_to_none_above_apex():
    # apex from y0=100 is at y = 100 - 75 = 25; a target above the apex (y<25)
    # is never reached on the descent.
    assert descent_time_to(100.0, 20.0, DYN) is None


def test_predict_descent_window_ordered():
    win = predict_descent_window(100.0, hole_top=60.0, hole_bottom=110.0, dyn=DYN)
    assert win is not None
    t_apex, t_in, t_out = win
    # apex first, then enter the hole top, then leave the hole bottom.
    assert t_apex < t_in < t_out


def test_crossing_window_arithmetic():
    # hoop left edge 300px ahead, 30px wide, approach 200px/s.
    t_enter, t_exit = crossing_window(0.0, 300.0, 330.0, DYN)
    assert t_enter == pytest.approx(1.5)
    assert t_exit == pytest.approx(1.65)


# --- firing decision ------------------------------------------------------

def test_should_flap_fires_when_hoop_near_not_far():
    # descent-to-centre time T ~= 0.947s (from descent_time_to(100,85)).
    # Far hoop (gap_left 300): t_mid 1.575s > T -> hold.
    assert should_flap_now(0, 100, 300, 330, 60, 110, DYN) is False
    # Near hoop (gap_left 150): t_mid 0.825s <= T -> fire.
    assert should_flap_now(0, 100, 150, 180, 60, 110, DYN) is True


def test_should_flap_monotonic_crossover():
    # As the hoop approaches (gap_left_x shrinks) the decision flips False->True
    # exactly once and stays True until the hoop passes.
    seen_true = False
    for gl in range(300, 10, -5):
        fire = should_flap_now(0, 100, gl, gl + 30, 60, 110, DYN)
        if fire:
            seen_true = True
        # once it fires it must not flip back to False while the hoop is ahead
        if seen_true and gl > 30:
            assert fire or (gl + 30) / DYN.approach_speed <= 0
    assert seen_true


def test_should_flap_false_without_hoop():
    assert should_flap_now(0, 100, None, None, None, None, DYN) is False


def test_should_flap_false_when_hoop_passed():
    # both edges already left of the avatar (negative crossing times).
    assert should_flap_now(100, 100, 40, 70, 60, 110, DYN) is False


def test_should_flap_false_when_flap_cannot_reach_hole():
    # avatar at y=100, apex at 25; a hole centred at y=15 is above the apex,
    # so a flap can't carry the descent into it.
    assert should_flap_now(0, 100, 150, 180, 10, 20, DYN) is False


# --- fit round-trip -------------------------------------------------------

def _synth_trace(dyn: Dynamics, dt=0.03, n=500, flap_period_s=0.55,
                 hoop_span=30.0, hoop_start=320.0, fly_x=0.0):
    """A dense trace from KNOWN dynamics: a hover policy flapping every
    flap_period_s (set-velocity), free fall between, and a hoop scrolling left
    at approach_speed (reset to the right when it passes the avatar). fly_vy is
    the finite difference of fly_y, as main.py computes it."""
    rows = []
    y, vy = 100.0, 0.0
    prev_y = None
    last_flap_t = -999.0
    gap_left = hoop_start
    for i in range(n):
        t = i * dt
        gl = gap_left - dyn.approach_speed * t
        if gl < fly_x - hoop_span:           # passed the avatar -> next hoop
            gap_left += hoop_start
            gl = gap_left - dyn.approach_speed * t
        fired = 0
        vy_used = vy
        if t - last_flap_t >= flap_period_s:
            vy_used = dyn.flap_vy
            fired = 1
            last_flap_t = t
        fly_vy = None if prev_y is None else (y - prev_y) / dt
        rows.append({
            "t": t, "fly_x": fly_x, "fly_y": y, "fly_vy": fly_vy,
            "gap_top": 60, "gap_bottom": 110,
            "gap_left_x": gl, "gap_right_x": gl + hoop_span,
            "gap_center": 85, "target_y": 90,
            "fired": fired, "where": "timed" if fired else "hover",
        })
        prev_y = y
        y, vy = step(y, vy_used, dt, dyn)
    return rows


def test_fit_recovers_known_dynamics():
    rows = _synth_trace(DYN)
    g = fit_gravity(rows)
    fv = fit_flap_vy(rows)
    ap = fit_approach_speed(rows)
    assert g == pytest.approx(DYN.gravity, rel=0.05)
    # observed launch velocity is the finite-diff average over the flap step,
    # ~flap_vy + 0.5 g dt, so allow a small slack.
    assert fv == pytest.approx(DYN.flap_vy, rel=0.08)
    assert ap == pytest.approx(DYN.approach_speed, rel=0.05)


def test_fit_dynamics_round_trip():
    dyn = fit_dynamics(_synth_trace(DYN))
    assert dyn is not None
    assert dyn.gravity == pytest.approx(DYN.gravity, rel=0.05)
    assert dyn.approach_speed == pytest.approx(DYN.approach_speed, rel=0.05)


def test_fit_dynamics_none_on_empty():
    assert fit_dynamics([]) is None


def test_dynamics_json_round_trip():
    d = DYN.to_dict()
    assert Dynamics.from_dict(d) == DYN
