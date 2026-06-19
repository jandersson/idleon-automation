"""Unit tests for the moving-target lead helpers (minigames/fishing/cast_model.py).

The targets slide back and forth (~cast 5+), so the bot leads the aim to the
fish's predicted landing position. These pin the two pure pieces: the robust
Theil-Sen velocity fit (which must beat the ±1px integer-centroid jitter and
refuse to lead at a turn/noise) and lead_fish_dist (turn-cap, reach-clamp, and
the no-abs-flip scalar-distance handling). The live velocity SAMPLING + A/B
wiring is in main.py and stays manual (CV against frames). See #58/#63.
"""
import pytest

from minigames.fishing.cast_model import (
    MAX_LEAD_PX, MIN_LEAD_VEL_PX_S, lead_fish_dist, theil_sen_velocity,
)


# --- lead_fish_dist ---------------------------------------------------------

WIDE = (50.0, 300.0)   # (reach_lo, reach_hi) wide enough not to clamp


def test_no_velocity_means_no_lead():
    # 3rd element is the clamp_reason (#85): "" = no lead / uncapped.
    assert lead_fish_dist(150, None, 1.3, *WIDE) == (150, 0.0, "")


def test_below_floor_velocity_means_no_lead():
    assert lead_fish_dist(150, MIN_LEAD_VEL_PX_S - 0.1, 1.3, *WIDE) == (150, 0.0, "")


def test_small_clean_lead_applies_exactly():
    # v=10 over 1.0s = 10px < the turn-cap -> applied as-is, no clamp reason.
    led, eff, reason = lead_fish_dist(150, 10.0, 1.0, *WIDE)
    assert led == 160 and eff == 10.0 and reason == ""


def test_outward_slide_is_turn_capped():
    # v=15 over 1.3s = 19.5px > MAX_LEAD_PX -> turn-capped, flagged "turn" (#85: the
    # old reach-only bool reported NOT-clamped here, hiding the cap the raise measures).
    led, eff, reason = lead_fish_dist(150, 15.0, 1.3, *WIDE)
    assert eff == MAX_LEAD_PX and led == 150 + MAX_LEAD_PX and reason == "turn"


def test_inward_slide_decreases_distance_no_abs_flip():
    # A fish sliding toward the origin (-v) must DECREASE target_dist, not flip it.
    led, eff, reason = lead_fish_dist(150, -10.0, 1.0, *WIDE)
    assert led == 140 and eff == -10.0 and reason == ""


def test_inward_slide_turn_capped_symmetric():
    led, eff, reason = lead_fish_dist(150, -22.0, 1.3, *WIDE)
    assert eff == -MAX_LEAD_PX and led == 150 - MAX_LEAD_PX and reason == "turn"


def test_reach_clamp_flags_and_trims_the_lead():
    # Near the far reach edge, the lead overshoots -> turn-cap AND reach-clamp fire.
    led, eff, reason = lead_fish_dist(295, 15.0, 1.3, 50.0, 300.0)
    assert led == 300 and reason == "turn+reach" and 0 < eff < MAX_LEAD_PX


def test_reach_clamp_only_when_under_the_turn_cap():
    # A small lead (under the turn-cap) that still overshoots reach -> "reach" alone.
    led, eff, reason = lead_fish_dist(298, 9.0, 1.0, 50.0, 300.0)   # raw 9 < cap, 298+9=307>300
    assert led == 300 and reason == "reach"


# --- theil_sen_velocity -----------------------------------------------------

def test_velocity_none_below_min_samples():
    assert theil_sen_velocity([(0.0, 100.0), (0.1, 102.0)]) is None


def test_velocity_none_when_span_too_short():
    # 3 samples but only 0.05s apart total -> can't resolve sub-pixel motion.
    assert theil_sen_velocity([(0.0, 100), (0.025, 100), (0.05, 101)]) is None


def test_velocity_recovers_clean_slope():
    v = theil_sen_velocity([(0.0, 100), (0.1, 102), (0.2, 104), (0.3, 106)])
    assert v == pytest.approx(20.0, abs=1e-6)


def test_velocity_robust_to_one_jittered_sample():
    # A +1px up-jitter that keeps consecutive deltas non-negative: the median
    # pairwise slope still sits near the true ~20 px/s (Theil-Sen robustness).
    v = theil_sen_velocity([(0.0, 100), (0.1, 103), (0.2, 104), (0.3, 106)])
    assert 18 <= v <= 32


def test_velocity_none_on_direction_reversal():
    # The fish reversed across the window (a turn / noise) -> refuse to lead.
    assert theil_sen_velocity([(0.0, 100), (0.1, 105), (0.2, 101)]) is None


def test_velocity_zero_for_stationary_fish():
    # Stationary -> ~0 velocity, which lead_fish_dist treats as no lead.
    v = theil_sen_velocity([(0.0, 120), (0.1, 120), (0.2, 120)])
    assert v == 0.0
    assert lead_fish_dist(120, v, 1.3, *WIDE) == (120, 0.0, "")
