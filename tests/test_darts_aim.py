"""Tests for the darts aim decision and the calm-air A/B selector (#48).

vy units: px/s (see minigames.darts.stripe_model). The A/B alternates the
static calm-air band against the wind-conditioned model in calm wind only,
tagged band_ab/model_ab so a powered band-vs-model comparison can later
accumulate under the same conditions.
"""
from minigames.darts.main import (
    CALM_WIND_MAX,
    MAX_AIM_SKIPS,
    VY_AIM_HI,
    VY_AIM_LO,
    _aim_fire_decision,
    _calm_ab_arm,
)


# --- _calm_ab_arm: deterministic calm-air split ---

def test_calm_ab_alternates_by_parity():
    """Even throw index -> band arm, odd -> model arm: ~50/50 and
    reconstructable from throw_idx alone."""
    band = (-44, -16)
    assert _calm_ab_arm(0, 0.0, band, False) == "band"   # parity starts at 0
    assert _calm_ab_arm(1, 0.0, band, False) == "model"
    assert _calm_ab_arm(2, 0.0, band, False) == "band"
    assert _calm_ab_arm(3, 0.0, band, False) == "model"
    assert _calm_ab_arm(4, 1.0, band, False) == "band"
    assert _calm_ab_arm(5, 1.0, band, False) == "model"
    # Pure + deterministic: identical args always pick the same arm, so a
    # throw's arm is stable across its multiple aiming polls.
    assert _calm_ab_arm(2, 0.0, band, False) == _calm_ab_arm(2, 0.0, band, False)


def test_calm_ab_inactive_at_or_above_calm_cut():
    """At/above CALM_WIND_MAX the A/B is off — the windy regime keeps the
    validated model path (#41) untouched."""
    band = (-44, -16)
    assert _calm_ab_arm(2, CALM_WIND_MAX, band, False) is None
    assert _calm_ab_arm(3, CALM_WIND_MAX + 5, band, False) is None
    # Just below the cut is still calm.
    assert _calm_ab_arm(2, CALM_WIND_MAX - 0.5, band, False) == "band"


def test_calm_ab_inactive_without_model_or_wind_or_on_explore():
    band = (-44, -16)
    assert _calm_ab_arm(2, 0.0, None, False) is None    # no model band fitted
    assert _calm_ab_arm(2, None, band, False) is None   # wind unparsed
    assert _calm_ab_arm(2, 0.0, band, True) is None      # exploration throw


# --- _aim_fire_decision: A/B arms fire/skip on their OWN band ---

def test_band_ab_fires_only_inside_static_band():
    """The band arm checks the static [-32,-20] band, not the model band:
    a vy that the (wide) model band would admit must not fire band_ab."""
    band = (-60, -8)  # wide model band that would otherwise admit vy=-12
    assert _aim_fire_decision(VY_AIM_LO, 0, False, band, "band") == (True, "band_ab")
    assert _aim_fire_decision(-28, 0, False, band, "band") == (True, "band_ab")
    assert _aim_fire_decision(VY_AIM_HI, 0, False, band, "band") == (True, "band_ab")
    assert _aim_fire_decision(-12, 0, False, band, "band") == (False, None)


def test_model_ab_fires_only_inside_model_band():
    """The model arm checks the model band, not the static band: a vy in
    the static band but outside the model band must not fire model_ab."""
    band = (-44, -36)  # narrow model band excluding the static [-32,-20]
    assert _aim_fire_decision(-40, 0, False, band, "model") == (True, "model_ab")
    assert _aim_fire_decision(-24, 0, False, band, "model") == (False, None)


def test_model_ab_needs_a_model_band():
    """Defensive: 'model' arm with no model band can't fire as model_ab."""
    assert _aim_fire_decision(-32, 0, False, None, "model") == (False, None)


def test_ab_arms_still_respect_the_skip_budget():
    """An out-of-band vy at the skip ceiling fires as 'fallback' (not the
    A/B tag) so the bot never starves — keeps the tags honest: only clean
    in-band fires carry band_ab/model_ab."""
    band = (-44, -36)
    assert _aim_fire_decision(-12, MAX_AIM_SKIPS, False, band, "band") == (True, "fallback")
    assert _aim_fire_decision(-12, MAX_AIM_SKIPS, False, band, "model") == (True, "fallback")
    # Tight boundary: one below the ceiling still waits, one above still
    # fires as fallback (never with an _ab tag).
    assert _aim_fire_decision(-12, MAX_AIM_SKIPS - 1, False, band, "band") == (False, None)
    assert _aim_fire_decision(-12, MAX_AIM_SKIPS + 1, False, band, "model") == (True, "fallback")


def test_explore_overrides_the_ab_arm():
    """Exploration fires on any valid pass regardless of the A/B arm, so
    the vy->outcome surface keeps getting sampled outside both bands."""
    band = (-44, -36)
    assert _aim_fire_decision(-12, 0, True, band, "band") == (True, "explore")
    assert _aim_fire_decision(-12, 0, True, band, "model") == (True, "explore")
    # ...but explore still needs a vy reading: explore + vy=None consumes
    # the skip budget rather than firing blind, and never tags an _ab.
    assert _aim_fire_decision(None, 0, True, band, "band") == (False, None)
    assert _aim_fire_decision(None, MAX_AIM_SKIPS, True, band, "model") == (True, "fallback")


# --- tag separability: the A/B's whole premise ---

def test_same_input_yields_distinct_organic_vs_ab_tags():
    """The A/B rests on the tags being separable in darts.db: the SAME
    (vy, model_band) must log 'model' organically but 'model_ab' under the
    model arm and 'band_ab' under the band arm. A regression that
    conflated these would silently corrupt the comparison with no other
    test catching it."""
    band = (-44, -20)
    vy = -30  # inside the model band AND inside the static [-32,-20] band
    assert _aim_fire_decision(vy, 0, False, band, None) == (True, "model")
    assert _aim_fire_decision(vy, 0, False, band, "model") == (True, "model_ab")
    assert _aim_fire_decision(vy, 0, False, band, "band") == (True, "band_ab")


# --- _aim_fire_decision: organic (non-A/B) paths unchanged ---

def test_model_path_unchanged_when_not_ab():
    band = (-44, -20)
    assert _aim_fire_decision(-32, 0, False, band, None) == (True, "model")
    assert _aim_fire_decision(-10, 0, False, band, None) == (False, None)


def test_static_band_path_when_no_model():
    assert _aim_fire_decision(VY_AIM_LO, 0, False, None, None) == (True, "band")
    assert _aim_fire_decision(VY_AIM_HI, 0, False, None, None) == (True, "band")
    assert _aim_fire_decision(0, 0, False, None, None) == (False, None)


def test_vy_none_consumes_skip_budget():
    """No vy reading: don't fire until the budget is exhausted (2026-06-11
    rule — blind fires hit ~3/10)."""
    band = (-44, -20)
    assert _aim_fire_decision(None, 0, False, band, None) == (False, None)
    assert _aim_fire_decision(None, MAX_AIM_SKIPS, False, band, None) == (True, "fallback")
    # Same under an A/B arm — a None vy never produces an _ab tag.
    assert _aim_fire_decision(None, 0, False, band, "band") == (False, None)
    assert _aim_fire_decision(None, MAX_AIM_SKIPS, False, band, "model") == (True, "fallback")
