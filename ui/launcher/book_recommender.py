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

Only talents we have metadata for are recommendable. The unmapped active
talents are the star/special block (save indices ~618+): those are raised
by **Special Talent Books** (drops / quests / Merit shops), a separate
system from the Library checkout pool (wiki: *Special Talents* /
*Talent Book Library*), so a Library recommendation for them would point
the user at a book that can't be checked out.
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
    KNOWN (present in `talent_meta` — see module docstring on why unmapped
    star talents are excluded), bookable (not on the can't-be-booked
    exclusion list), currently AT CAP (`level >= max`), and a higher book
    exists (`max < max_book_level`).

    Returns a list of dicts (best first):
        {index, name, klass, level, cap, gap, importance}
    where gap = max_book_level - cap and `klass` is the talent's class line
    (Barbarian, Wizard, Beginner, …) — the Library "subgenre". The Books tab
    pairs it with its base class (talent_data.base_genre) to show the full
    genre › subgenre navigation path to the book.
    """
    out: list[dict] = []
    n = min(len(skill_levels), len(skill_levels_max))
    for i in range(n):
        cap = skill_levels_max[i]
        lvl = skill_levels[i]
        if cap == INACTIVE or cap is None or lvl is None:
            continue
        meta = talent_meta.get(i)
        if meta is None:
            continue  # unmapped → star/special talent, not Library-bookable
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


def recommend_books_account(
    talents: dict[str, dict],
    talent_meta: dict[int, dict],
    max_book_level: int,
) -> list[dict]:
    """Account-wide book candidates ranked across ALL characters.

    Library checkouts come from one shared pool (OLA[55]), so the real
    question isn't "what's best for this character" but "where do my next
    checkouts go" across the whole account. Flattens every character's
    per-character candidates into one list, tags each with its
    `character`, and ranks by the same (importance, -gap, -cap) key.

    `talents` is the `read_talents()` shape: {char_name: {"skill_levels",
    "skill_levels_max", "character_class"}}.
    """
    out: list[dict] = []
    for name, t in talents.items():
        for c in recommend_books(
            t.get("skill_levels") or [],
            t.get("skill_levels_max") or [],
            talent_meta,
            max_book_level,
        ):
            out.append({**c, "character": name})
    out.sort(key=lambda c: (c["importance"], -c["gap"], -c["cap"]))
    return out
