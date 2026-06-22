"""Tests for the stamp recommender (discovery map: Stamps) — pure logic.

Cost values are hand-checked against the IdleonToolbox-ported formulas; the
ranking is checked on a synthetic stamp table + owned-material map.
"""
from ui.launcher.stamp_recommender import gold_cost, material_cost, recommend_stamps


def test_material_cost_hand_checked():
    s = {"base_mat": 2, "pow_mat": 3, "req_mult": 1}
    assert material_cost(1, s) == 2          # step 0 -> 3**0 = 1 -> 2
    # level 5: step = round(5/1)-1 = 4; 4**0.8 ≈ 3.031; 3**3.031 ≈ 27.93; *2 -> 55
    assert material_cost(5, s) == 55


def test_gold_cost_hand_checked():
    s = {"base_coin": 10, "pow_coin": 1.2, "req_mult": 1}
    # level 1: pow_base = max(1.05, 1.2-(1/6)*0.25) ≈ 1.158; 10 * 1.158**10 ≈ 43
    assert gold_cost(1, s) == 43


def test_material_cost_floors_at_one():
    assert material_cost(1, {"base_mat": 0, "pow_mat": 5, "req_mult": 10}) == 1


META = {
    "combat": [
        {"name": "A", "base_coin": 10, "pow_coin": 1.2, "base_mat": 2, "pow_mat": 3,
         "req_mult": 1, "mat_raw": "matA", "mat_name": "Mat A"},
        {"name": "B", "base_coin": 10, "pow_coin": 1.2, "base_mat": 5, "pow_mat": 3,
         "req_mult": 1, "mat_raw": "matB", "mat_name": "Mat B"},
    ],
    "skills": [
        {"name": "C", "base_coin": 10, "pow_coin": 1.2, "base_mat": 2, "pow_mat": 3,
         "req_mult": 1, "mat_raw": "matC", "mat_name": "Mat C"},
    ],
    "misc": [],
}


def test_recommend_ranks_affordable_then_cheapest():
    # A@5 -> 55 matA (have 100, affordable); B@5 -> 139 matB (have 0, not);
    # C@1 -> 2 matC (have 10, affordable). Affordable-first, cheapest-first.
    levels = {"combat": [5, 5], "skills": [1], "misc": []}
    owned = {"matA": 100, "matB": 0, "matC": 10}
    recs = recommend_stamps(levels, owned, META)
    assert [r["name"] for r in recs] == ["C", "A", "B"]
    assert recs[0]["affordable"] is True and recs[0]["mat_cost"] == 2
    assert recs[-1]["name"] == "B" and recs[-1]["affordable"] is False


def test_recommend_skips_level_zero_stamps():
    # B and C are at level 0 -> not owned/leveled -> excluded.
    levels = {"combat": [5, 0], "skills": [0], "misc": []}
    recs = recommend_stamps(levels, {"matA": 100}, META)
    assert [r["name"] for r in recs] == ["A"]


def test_recommend_top_n_limits():
    levels = {"combat": [5, 5], "skills": [1], "misc": []}
    recs = recommend_stamps(levels, {"matA": 100, "matC": 10}, META, top_n=2)
    assert len(recs) == 2
