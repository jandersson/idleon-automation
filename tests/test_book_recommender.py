"""Tests for the talent-book recommender (#44) — pure logic, synthetic data."""
from ui.launcher.book_recommender import recommend_books

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


def test_unknown_index_defaults_bookable_and_named():
    # An at-cap active talent with no meta entry is still surfaced
    # (named by index, default importance) — better to over-list than drop.
    recs = recommend_books([100], [100], {}, MAXBOOK)
    assert len(recs) == 1
    assert recs[0]["name"] == "#0" and recs[0]["importance"] == 3
