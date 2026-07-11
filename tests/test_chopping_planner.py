"""Planned-shot aiming (minigames/chopping/planner.py).

Pure-model tests: zone parsing, target choice, margin rejection,
direction mirroring, cooldown shifting, and consistency between the
eased time and position mappings. Bar geometry mirrors the live one
(222px wide, V_max 300-650 px/s).
"""
import math

from minigames.chopping.detector import eased_time_to_red_ms, eased_time_to_x_s
from minigames.chopping.planner import Plan, Zone, parse_layout, plan_shot

BAR_W = 222


def test_parse_layout_spans():
    zones = parse_layout("n4 g58 r12 o9 g70 n69")
    assert [z.kind for z in zones] == ["n", "g", "r", "o", "g", "n"]
    assert zones[0].x0 == 0 and zones[0].x1 == 4
    assert zones[3].x0 == 74 and zones[3].x1 == 83
    assert zones[-1].x1 == 222


def test_eased_time_to_red_ms_wraps_time_to_x():
    t_s = eased_time_to_x_s(50, 1.0, 60, 500.0, BAR_W)
    t_ms = eased_time_to_red_ms(50, 1.0, 60, 500.0, BAR_W)
    assert t_ms == int(t_s * 1000)


def test_plans_the_green_center_when_only_green_ahead():
    # leaf at 10 moving right; green 40..140, red at both bar ends.
    zones = parse_layout("r20 n20 g100 n42 r40")
    plan = plan_shot(zones, 10.0, 1.0, 400.0, BAR_W)
    assert plan is not None and plan.zone_kind == "g" and plan.value == 1
    assert 40 < plan.target_x < 140
    assert plan.margin_ms >= 48  # 3 sigma at the default 16ms


def test_prefers_gold_over_green_when_feasible():
    zones = parse_layout("n10 g60 n30 o40 n42 r40")
    plan = plan_shot(zones, 5.0, 1.0, 350.0, BAR_W)
    assert plan is not None
    assert plan.zone_kind == "o" and plan.value == 2
    assert 100 < plan.target_x < 140


def test_red_flanked_gold_rejected_at_speed_but_green_taken():
    # The 21:01 run's doomed shape: r | g | r o r | g-ish tail.
    zones = parse_layout("r40 g80 r10 o20 r30 g42")
    plan = plan_shot(zones, 45.0, 1.0, 650.0, BAR_W)
    assert plan is not None
    # 20px of red-sandwiched gold at 650 px/s can't give 3x16ms margins.
    assert plan.zone_kind == "g"


def test_mirrors_for_leftward_motion():
    zones = parse_layout("r40 g100 n42 r40")
    right = plan_shot(zones, 45.0, 1.0, 400.0, BAR_W)
    zones_m = parse_layout("r40 n42 g100 r40")
    left = plan_shot(zones_m, 177.0, -1.0, 400.0, BAR_W)
    assert right is not None and left is not None
    # Mirror symmetry: targets reflect around the bar center.
    assert abs((BAR_W - left.target_x) - right.target_x) < 3.0
    assert abs(left.impact_in_s - right.impact_in_s) < 0.01


def test_earliest_impact_shifts_target_later():
    zones = parse_layout("n10 g100 n72 r40")
    base = plan_shot(zones, 5.0, 1.0, 400.0, BAR_W)
    shifted = plan_shot(zones, 5.0, 1.0, 400.0, BAR_W,
                        earliest_impact_s=base.impact_in_s + 0.05)
    assert shifted is not None
    assert shifted.impact_in_s >= base.impact_in_s + 0.05
    assert shifted.target_x > base.target_x


def test_earliest_impact_beyond_zone_rejects():
    # Cooldown wants an impact after the whole crossing — no plan.
    zones = parse_layout("n10 g50 r162")
    plan = plan_shot(zones, 5.0, 1.0, 500.0, BAR_W,
                     earliest_impact_s=5.0)
    assert plan is None


def test_zone_behind_is_planned_on_the_next_sweep():
    # Leaf moving right with nothing plannable ahead — pre-cross-sweep
    # this returned None; now the mid-bar green behind is planned
    # through the bounce.
    zones = parse_layout("n30 g80 n112")
    plan = plan_shot(zones, 120.0, 1.0, 400.0, BAR_W)
    assert plan is not None
    assert plan.sweep == 1
    assert plan.zone_kind == "g" and 30 < plan.target_x < 110


def test_next_sweep_impact_time_includes_the_bounce():
    # The same next-sweep target must land AFTER the turnaround.
    zones = parse_layout("n30 g80 n112")
    plan = plan_shot(zones, 120.0, 1.0, 400.0, BAR_W)
    t_to_turn = eased_time_to_x_s(120, 1.0, BAR_W - 120, 400.0, BAR_W)
    assert plan.impact_in_s > t_to_turn


def test_same_sweep_beats_next_sweep_at_equal_geometry():
    # A symmetric green ahead and behind: the near-sweep candidate has
    # the smaller horizon (smaller sigma), so it must win.
    zones = parse_layout("g60 n51 g60 n51")
    plan = plan_shot(zones, 105.0, 1.0, 400.0, BAR_W)
    assert plan is not None
    assert plan.sweep == 0
    assert plan.target_x > 105


def test_turnaround_red_pocket_rejected_from_both_sides():
    # Red slivers at the far end are crossed twice around the bounce;
    # both crossing times must suppress next-sweep impacts near them.
    zones = parse_layout("n1 g150 r10 o50 r10 n1")
    plan = plan_shot(zones, 140.0, 1.0, 600.0, BAR_W, red_sigmas=4.0)
    # Gold sits between two reds against the turnaround: not plannable
    # at speed from either sweep — the green must win if anything does.
    assert plan is None or plan.zone_kind == "g"


def test_no_plan_without_speed():
    zones = parse_layout("n10 g100 n112")
    assert plan_shot(zones, 5.0, 1.0, 0.0, BAR_W) is None
    assert plan_shot(zones, 5.0, 0.0, 400.0, BAR_W) is None


def test_edge_targets_need_scaled_margins_chop18_regression():
    """The 23:44 run's death: gold at 180-214 with turnaround red
    slivers past it, leaf at 173 rightward, ~700 px/s V_max. The flat
    3-sigma margin accepted a 51ms plan at sin(theta)=0.61; the
    position-scaled budget must reject any plan into that pocket."""
    zones = parse_layout("n1 r94 g85 o34 r2 g3 r2 n1")
    plan = plan_shot(zones, 173.0, 1.0, 700.0, BAR_W,
                     earliest_impact_s=0.055, red_sigmas=4.0)
    assert plan is None or plan.zone_kind != "o" or plan.target_x < 180


def test_mid_bar_green_is_planned_with_ample_margin():
    # Red receding behind, none ahead for a long time: any in-zone
    # impact is fine and the planner may fire early (higher chop rate).
    zones = parse_layout("r40 g100 n42 r40")
    plan = plan_shot(zones, 45.0, 1.0, 400.0, BAR_W)
    assert plan is not None
    assert 45 < plan.target_x < 140  # inside the green, ahead of the leaf
    assert plan.margin_ms > 100
