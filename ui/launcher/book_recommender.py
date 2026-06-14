"""Talent-book recommendation logic (#44) — pure, no Tk / no IO.

Given a character's talent levels + caps (from the save) and the talent
metadata map, rank which talents would benefit most from checking out a
higher Library book.

A book raises ONE talent's max level (per-talent, not global). The
recommendable signal is entirely `SkillLevelsMAX` vs `SkillLevels`:
- A talent AT CAP (level >= max) is fully invested — a higher book would
  immediately buy more usable levels.
- A talent with headroom (level < max) should be filled with talent
  points first; a book wouldn't help yet, so it's not recommended.

Only talents whose current cap is below the account `max_book_level` have
any room for a better book (`gap = max_book_level - cap > 0`).

Candidates are ranked by talent importance (1 = highest) then by the gap
(how many cap-levels a fresh max book unlocks).
"""
from __future__ import annotations

INACTIVE = -1  # SkillLevelsMAX sentinel for "not a real/owned talent"
DEFAULT_IMPORTANCE = 3


def recommend_books(
    skill_levels: list[int],
    skill_levels_max: list[int],
    talent_meta: dict[int, dict],
    max_book_level: int,
) -> list[dict]:
    """Ranked book candidates for one character.

    A talent index is a candidate iff it is active (`max != INACTIVE`),
    bookable (not on the can't-be-booked exclusion list), currently AT CAP
    (`level >= max`), and a higher book exists (`max < max_book_level`).

    Returns a list of dicts (best first):
        {index, name, klass, level, cap, gap, importance}
    where gap = max_book_level - cap.
    """
    out: list[dict] = []
    n = min(len(skill_levels), len(skill_levels_max))
    for i in range(n):
        cap = skill_levels_max[i]
        lvl = skill_levels[i]
        if cap == INACTIVE or cap is None or lvl is None:
            continue
        meta = talent_meta.get(i, {})
        if meta.get("bookable", True) is False:
            continue
        if lvl < cap:
            continue  # headroom remains — fill with points first
        gap = max_book_level - cap
        if gap <= 0:
            continue  # already at the account's max book level
        out.append({
            "index": i,
            "name": meta.get("name", f"#{i}"),
            "klass": meta.get("class"),
            "level": lvl,
            "cap": cap,
            "gap": gap,
            "importance": meta.get("importance", DEFAULT_IMPORTANCE),
        })
    out.sort(key=lambda c: (c["importance"], -c["gap"], -c["cap"]))
    return out
