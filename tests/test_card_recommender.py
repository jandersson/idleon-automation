"""Tests for the card star recommender (cards/stamps step).

The star math is a verbatim port of IdleonToolbox parsers/cards.ts, so
the threshold cases below double as a port-fidelity check. The save
readers (counts / five-star list / star cap) are monkeypatched onto
load_save since they normally need a live LevelDB save.
"""
import common.idleon_save as save
from common.idleon_save import (
    read_card_counts,
    read_card_five_star_set,
    read_card_star_cap,
    _OLA_FIVE_STAR_CARDS,
)
from ui.launcher.card_recommender import (
    calculate_stars,
    copies_to_next_star,
    recommend_cards,
)


# ---- star math (port fidelity) ---------------------------------------

def test_calculate_stars_threshold_sequence():
    # per_tier=10: star i reached when amount > 10*(i+1+...)^2.
    # thresholds: 1*=10, 2*=40, 3*=160, 4*=250  (exclusive)
    f = lambda amt: calculate_stars(10, amt, "x", 5, False)
    assert f(10) == 0      # not > 10
    assert f(11) == 1
    assert f(40) == 1      # not > 40
    assert f(41) == 2
    assert f(161) == 3
    assert f(251) == 4


def test_calculate_stars_five_star_list_pins_to_5():
    # In the five-star list and below 6 → auto 5 stars.
    assert calculate_stars(10, 5, "x", 5, True) == 5


def test_calculate_stars_boss3b_special_case():
    # Boss3B uses 1.5*(i+1+floor(i/3))^2 instead of the perTier formula.
    assert calculate_stars(999, 2, "Boss3B", 5, False) == 1   # 2 > 1.5
    assert calculate_stars(999, 1, "Boss3B", 5, False) == 0   # 1 !> 1.5


def test_copies_to_next_star():
    # 0★, 5 copies, per_tier 10 → need >10, i.e. 6 more (5+6=11).
    assert copies_to_next_star(10, 0, 5) == 6
    # 1★, 11 copies → next threshold 40, need 30 more (11+30=41).
    assert copies_to_next_star(10, 1, 11) == 30
    # at the 6-star ceiling → nothing more.
    assert copies_to_next_star(10, 6, 10_000) == 0


# ---- recommend_cards --------------------------------------------------

META = {
    "a": {"name": "Aye", "per_tier": 10, "effect": "+x", "set": "S1"},
    "near": {"name": "Near", "per_tier": 1, "effect": "+y", "set": "S1"},
    "maxed": {"name": "Maxed", "per_tier": 1, "effect": "+z", "set": "S2"},
    "five": {"name": "Fived", "per_tier": 10, "effect": "+f", "set": "S2"},
}


def test_recommend_filters_and_ranks():
    counts = {
        "a": 11,         # 1★, needs 30 → candidate
        "near": 2,       # per_tier 1: 1★ at >1, 2★ at >4 → needs 3 → candidate (closest)
        "maxed": 10_000,  # way past 5★ cap → excluded
        "five": 5,       # in five-star list → excluded
        "unknown": 50,   # not in META → excluded
        "zero": 0,       # not owned → excluded
    }
    recs = recommend_cards(counts, META, max_stars=5, five_star_set={"five"})
    codes = [r["code"] for r in recs]
    assert codes == ["near", "a"]            # ranked by fewest copies-to-next
    assert recs[0]["to_next"] == 3 and recs[0]["stars"] == 1
    assert all(r["code"] not in ("maxed", "five", "unknown", "zero") for r in recs)


def test_recommend_empty_when_nothing_below_cap():
    assert recommend_cards({"maxed": 10_000}, META, max_stars=5) == []


# ---- save readers (monkeypatched) ------------------------------------

def test_read_card_counts(monkeypatch):
    monkeypatch.setattr(save, "load_save",
                        lambda save_dir=None: {"Cards": [{"snowball": 576, "x": 0}, [], [], {}]})
    assert read_card_counts() == {"snowball": 576, "x": 0}


def test_read_card_counts_none_on_bad_shape(monkeypatch):
    monkeypatch.setattr(save, "load_save", lambda save_dir=None: {"Cards": []})
    assert read_card_counts() is None
    monkeypatch.setattr(save, "load_save", lambda save_dir=None: None)
    assert read_card_counts() is None


def test_read_five_star_set(monkeypatch):
    ola = [0] * (_OLA_FIVE_STAR_CARDS + 1)
    ola[_OLA_FIVE_STAR_CARDS] = "snowball,crabcake"
    monkeypatch.setattr(save, "load_save", lambda save_dir=None: {"OptionsListAccount": ola})
    assert read_card_five_star_set() == {"snowball", "crabcake"}
    # default (integer 0, the unset state) → empty set, not {"0"} junk.
    ola[_OLA_FIVE_STAR_CARDS] = 0
    assert read_card_five_star_set() == set()


def test_read_card_star_cap(monkeypatch):
    monkeypatch.setattr(save, "load_save", lambda save_dir=None: {"Rift": [50]})
    assert read_card_star_cap() == 5          # Rift level >= 45 → +1
    monkeypatch.setattr(save, "load_save", lambda save_dir=None: {"Rift": [0]})
    assert read_card_star_cap() == 4
    monkeypatch.setattr(save, "load_save", lambda save_dir=None: {})
    assert read_card_star_cap() == 4          # no Rift → base
