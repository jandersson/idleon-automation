"""Tests for the talent-book recommender (#44) — pure logic, synthetic data."""
from ui.launcher.book_recommender import recommend_books, recommend_books_account
from ui.launcher.talent_data import CLASS_BY_ID, class_name

# index -> meta. Importance 1 = highest.
META = {
    0: {"name": "Gilded Sword", "class": "Warrior", "importance": 1, "bookable": True},
    1: {"name": "Fist of Rage", "class": "Warrior", "importance": 1, "bookable": False},
    2: {"name": "Mana Booster", "class": "Mage", "importance": 4, "bookable": True},
    3: {"name": "Weapon Power", "class": "Warrior", "importance": 2, "bookable": True},
}
MAXBOOK = 396


def _rec(levels, caps, meta=META, maxbook=MAXBOOK):
    return recommend_books(levels, caps, meta, maxbook)


def test_at_cap_bookable_talents_are_candidates():
    # idx0 at cap (100/100) → candidate; idx2 has headroom (50/100) → not.
    recs = _rec([100, 0, 50, 100], [100, -1, 100, 100])
    idxs = [r["index"] for r in recs]
    assert 0 in idxs and 3 in idxs
    assert 2 not in idxs  # headroom remains → fill with points first


def test_excludes_inactive_and_non_bookable():
    # idx1 (Fist of Rage, bookable=False) at cap must be excluded;
    # idx2 inactive (max=-1) excluded.
    recs = _rec([100, 100, 100, 100], [100, 100, -1, 100])
    idxs = [r["index"] for r in recs]
    assert 1 not in idxs
    assert 2 not in idxs


def test_no_candidate_when_cap_already_at_max_book():
    # cap == max_book_level → gap 0 → no better book exists.
    recs = _rec([396, 0, 0, 0], [396, -1, -1, -1])
    assert recs == []


def test_ranked_by_importance_then_gap():
    # idx0 (imp1, cap 100, gap 296) should outrank idx3 (imp2, cap 100).
    recs = _rec([100, 0, 0, 100], [100, -1, -1, 100])
    assert [r["index"] for r in recs] == [0, 3]


def test_gap_is_max_book_minus_cap():
    recs = _rec([120, 0, 0, 0], [120, -1, -1, -1])
    assert recs[0]["gap"] == MAXBOOK - 120


def test_unmapped_index_is_skipped():
    # An at-cap active talent with NO meta entry is the star/special block,
    # which isn't Library-bookable — it must be skipped, not surfaced.
    assert recommend_books([100], [100], {}, MAXBOOK) == []
    # Even alongside a known candidate, the unmapped one is dropped.
    recs = recommend_books([100, 100], [100, 100], {0: META[0]}, MAXBOOK)
    assert [r["index"] for r in recs] == [0]


def test_account_ranking_flattens_and_tags_characters():
    talents = {
        "Warr": {"skill_levels": [100, 0, 0, 100], "skill_levels_max": [100, -1, -1, 100],
                 "character_class": 8},
        "Mage": {"skill_levels": [0, 0, 100, 0], "skill_levels_max": [-1, -1, 100, -1],
                 "character_class": 31},
    }
    recs = recommend_books_account(talents, META, MAXBOOK)
    # Warr idx0 (imp1) > Warr idx3 (imp2) > Mage idx2 (imp4).
    assert [(r["character"], r["index"]) for r in recs] == [
        ("Warr", 0), ("Warr", 3), ("Mage", 2),
    ]
    assert all("character" in r for r in recs)


def test_account_ranking_empty_when_nothing_at_cap():
    talents = {"A": {"skill_levels": [50], "skill_levels_max": [100], "character_class": 8}}
    assert recommend_books_account(talents, META, MAXBOOK) == []


def test_class_name_maps_known_ids_and_degrades_gracefully():
    # Verified against the local save's characters.
    assert CLASS_BY_ID[8] == "Barbarian"
    assert CLASS_BY_ID[9] == "Squire"
    assert CLASS_BY_ID[20] == "Bowman"
    assert CLASS_BY_ID[21] == "Hunter"
    assert CLASS_BY_ID[31] == "Mage"
    assert CLASS_BY_ID[32] == "Wizard"   # Mage line breaks the W2/W3 ordering
    assert CLASS_BY_ID[33] == "Shaman"
    assert class_name(8) == "Barbarian"
    assert class_name(999) == "Class 999"  # unknown id → readable fallback
    assert class_name(None) == "?"
