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
    hover_target_y,
    floor_rescue_due,
    coast_rescue_due,
    plan_flap,
    SpeedTracker,
    RingProjector,
    fit_gravity,
    fit_flap_vy,
    fit_approach_speed,
    fit_dynamics,
)


DYN = Dynamics(gravity=600.0, flap_vy=-300.0, approach_speed=200.0)

# The live fitted dynamics (2026-06-17 trace) — the planner tests use these
# so the scenarios match the real geometry: rise ~44px, apex ~0.477s.
PDYN = Dynamics(gravity=388.0, flap_vy=-185.0, approach_speed=78.0)
PLAY_H = 185.0
FLY_X = 256.0


def _ring(t_mid, hc, speed=78.0, half_w=8.5, band_half=26.0):
    """(gl, gr, gt, gb) for a ring whose centre crosses FLY_X in t_mid s."""
    cx = FLY_X + t_mid * speed
    return (cx - half_w, cx + half_w, hc - band_half, hc + band_half)


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


# --- firing decision (apex-threading) -------------------------------------
# DYN: rise = 75px, time_to_apex = 0.5s, approach = 200px/s. With fly_x=0 the
# apex lands within the crossing when the hoop spans gap_left<=100<=gap_right
# (t_apex=0.5s in [gl/200, gr/200]); the apex lands in the hole when
# fly_y-75 is inside [hole_top, hole_bottom].

def test_should_flap_fires_when_apex_lands_in_hole_at_crossing():
    # hoop spanning the apex-time crossing (gl=90,gr=110), avatar at a launch
    # height whose apex (160-75=85) is in the hole [60,140].
    assert should_flap_now(0, 160, 90, 110, 60, 140, DYN) is True


def test_should_flap_false_when_hoop_too_far():
    # gl=200 -> t_enter=1.0s, apex(0.5s) lands BEFORE the hoop arrives.
    assert should_flap_now(0, 160, 200, 220, 60, 140, DYN) is False


def test_should_flap_false_when_hoop_nearly_passed():
    # gr=30 -> t_exit=0.15s, apex(0.5s) lands AFTER the hoop is gone.
    assert should_flap_now(0, 160, 10, 30, 60, 140, DYN) is False


def test_should_flap_false_when_apex_overshoots_hole_top():
    # right timing (gl=90,gr=110) but avatar too high: apex 100-75=25 is above
    # the hole top -> the flap would carry the avatar over the ring.
    assert should_flap_now(0, 100, 90, 110, 60, 140, DYN) is False


def test_should_flap_false_when_apex_undershoots_hole():
    # right timing but avatar too low: apex 230-75=155 is below the hole bottom.
    assert should_flap_now(0, 230, 90, 110, 60, 140, DYN) is False


def test_should_flap_fires_in_a_bounded_window_only():
    # As the hoop approaches with the avatar held at launch height, the launch
    # fires only in a bounded window (apex within the crossing), not forever.
    fires = [gl for gl in range(220, 0, -5)
             if should_flap_now(0, 160, gl, gl + 20, 60, 140, DYN)]
    assert fires, "should fire somewhere"
    # all firing gap_left_x are clustered where the apex-time lands in-crossing
    assert max(fires) <= 105 and min(fires) >= 75


def test_should_flap_false_without_hoop():
    assert should_flap_now(0, 160, None, None, None, None, DYN) is False


# --- hover target + floor rescue ------------------------------------------

def test_hover_target_is_bob_bottom():
    # apex on the hole centre -> hover (bob bottom) one rise below it.
    assert hover_target_y(85, DYN) == pytest.approx(85 + DYN.rise_height_px)


def test_floor_rescue_fires_below_bound():
    # bound = 0.70 * 185 = 129.5
    assert floor_rescue_due(150, 0, 185) is True
    assert floor_rescue_due(100, 0, 185) is False


def test_floor_rescue_is_predictive_on_fast_descent():
    # 120 < 129.5, but 120 + 300*0.1 = 150 crosses the bound within lookahead.
    assert floor_rescue_due(120, 300, 185, lookahead_s=0.1) is True
    # rising avatar never triggers it
    assert floor_rescue_due(80, -200, 185) is False


def test_floor_rescue_handles_missing_velocity():
    assert floor_rescue_due(100, None, 185) is False


def test_coast_rescue_fires_only_on_measured_descent():
    # below the coast floor AND descending -> rescue
    assert coast_rescue_due(120, 150, 110) is True
    # below the floor but RISING (fresh launch flap in flight) must NOT
    # re-flap — the ascent-blind rescue over-lifted the apex into the top
    # rim (run 16's death; #62 sim replay).
    assert coast_rescue_due(120, -150, 110) is False
    # unknown velocity (detection-gap frame) never rescues; floor_rescue_due
    # backstops a genuine sink on position alone.
    assert coast_rescue_due(120, None, 110) is False
    # above the floor: nothing to rescue.
    assert coast_rescue_due(100, 150, 110) is False


# --- phase-locked planner (#61) --------------------------------------------

def test_plan_flap_never_flaps_while_ascending():
    for t_mid in (0.3, 0.55, 1.0, 3.0):
        gl, gr, gt, gb = _ring(t_mid, 80)
        do, mode, _ = plan_flap(120.0, -200.0, gl, gr, gt, gb,
                                FLY_X, PLAY_H, PDYN, speed=78.0)
        assert do is False, f"flapped while ascending (t_mid={t_mid}, {mode})"


def test_plan_flap_launches_at_anchor_on_time():
    # Avatar descending at flap speed, positioned so the flap LANDS on the
    # anchor (hole_centre + rise) exactly one apex-time + latency before
    # the crossing: the definition of a clean launch.
    hc = 80.0
    anchor = hc + PDYN.rise_height_px
    lat = 0.075
    fly_y = anchor - (185.0 * lat + 0.5 * PDYN.gravity * lat * lat)
    t_mid = lat + PDYN.time_to_apex_s
    gl, gr, gt, gb = _ring(t_mid, hc)
    do, mode, _ = plan_flap(fly_y, 185.0, gl, gr, gt, gb,
                            FLY_X, PLAY_H, PDYN, speed=78.0)
    assert (do, mode) == (True, "launch")


def test_plan_flap_dodges_when_height_missed():
    # Same crossing timing but the avatar is far too high for the apex to
    # land in the hole — the launch must refuse and, with the window this
    # close, the planner holds an under-dodge (no flap from mid-sky).
    hc = 80.0
    t_mid = 0.075 + PDYN.time_to_apex_s
    gl, gr, gt, gb = _ring(t_mid, hc)
    do, mode, _ = plan_flap(80.0, 185.0, gl, gr, gt, gb,
                            FLY_X, PLAY_H, PDYN, speed=78.0)
    assert mode == "dodge_under"
    assert do is False


def test_plan_flap_prepositions_over_deep_ring():
    # A ring whose anchor sits below the survivable floor can't be
    # threaded — the planner climbs above its band ahead of time, and the
    # over-dodge trigger doubles as the climb driver from below.
    gl, gr, gt, gb = _ring(3.0, 120.0)
    do, mode, _ = plan_flap(130.0, 100.0, gl, gr, gt, gb,
                            FLY_X, PLAY_H, PDYN, speed=78.0)
    assert mode == "dodge_over"
    assert do is True


def test_plan_flap_waits_without_ring():
    do, mode, frac = plan_flap(100.0, 100.0, None, None, None, None,
                               FLY_X, PLAY_H, PDYN)
    assert (do, mode, frac) == (False, "wait", None)


def test_speed_tracker_recovers_linear_scroll():
    tr = SpeedTracker()
    t = 0.0
    while t <= 1.4:
        tr.add(t, 500.0 - 90.0 * t)
        t += 0.05
    assert tr.speed() == pytest.approx(90.0, rel=0.02)
    # a rightward jump = the next hoop became the target: window resets
    tr.add(1.45, 500.0)
    assert tr.speed() is None


def test_ring_projector_projects_and_expires():
    proj = RingProjector(fly_x=FLY_X, fallback_speed=80.0)
    proj.update(0.0, 400.0, 417.0, 54.0, 106.0)
    got = proj.get(0.2)
    assert got is not None
    assert got[0] == pytest.approx(400.0 - 80.0 * 0.2)
    assert got[2:] == (54.0, 106.0)
    assert proj.get(0.8) is None  # stale


def test_ring_projector_holds_ring_through_overlap():
    # The detector switches to the NEXT ring while the old one still
    # x-overlaps the avatar; the projector must keep the old ring until it
    # clears, or the planner flaps through its rim.
    proj = RingProjector(fly_x=FLY_X, fallback_speed=80.0)
    proj.update(0.0, 250.0, 267.0, 54.0, 106.0)   # old ring overlapping
    proj.update(0.05, 310.0, 327.0, 30.0, 82.0)   # detector jumped ahead
    got = proj.get(0.1)
    assert got is not None
    assert got[0] < 260.0  # still the old ring's projection


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
    fv = fit_flap_vy(rows, gravity=DYN.gravity)  # apex-rise needs g
    ap = fit_approach_speed(rows)
    assert g == pytest.approx(DYN.gravity, rel=0.05)
    # apex-rise recovers the set velocity tightly on clean synthetic arcs.
    assert fv == pytest.approx(DYN.flap_vy, rel=0.05)
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
