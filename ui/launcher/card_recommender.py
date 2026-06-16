"""Card star recommender — pure, no Tk / no IO.

The "book recommender for cards": given each card's collected copy count
(save `Cards[0]`) and the card metadata (`card_data.CARD_META`), surface
the cards CLOSEST to their next star — the high-ROI farming targets ("a
few more X for ★→★+1").

The star math is ported verbatim from IdleonToolbox `parsers/cards.ts`
(`calculateStars` / `calculateAmountToNextLevel`) so the numbers match
what players see in that tool:

    stars(i reached) iff amount > perTier * (i+1 + floor(i/3)
                          + 16*floor(i/4) + 100*floor(i/5))**2

`max_stars` is the account star cap (4 normally; +1 once Rift level >= 45,
+1 once the relevant Spelunking lore boss is beaten -> up to 6). Cards in
the account "five-star list" are auto-maxed and excluded. A card whose
stars already equal the cap has no next star and isn't recommended.
"""
from __future__ import annotations

import math

BASE_MAX_STARS = 4
HARD_STAR_CAP = 6  # the absolute ceiling the formula tops out at


def calculate_stars(
    per_tier: int,
    amount: int,
    card_name: str,
    max_stars: int,
    is_five_star: bool,
) -> int:
    """Current star level for a card (0-indexed, matching IdleonToolbox).

    Ported from `calculateStars` in IdleonToolbox parsers/cards.ts —
    including the 'Boss3B' special case (1.5*(i+1+floor(i/3))**2) and the
    five-star-list override that pins a card to 5."""
    card_lv = 0
    for i in range(max_stars):
        if card_name == "Boss3B":
            if amount > 1.5 * (i + 1 + i // 3) ** 2:
                card_lv = i + 2
        else:
            thresh = per_tier * (i + 1 + (i // 3 + 16 * (i // 4) + 100 * (i // 5))) ** 2
            if amount > thresh:
                card_lv = i + 2
    if is_five_star and card_lv < 6:
        return 5
    return card_lv - 1 if card_lv > 0 else card_lv


def copies_to_next_star(per_tier: int, stars: int, amount: int) -> int:
    """Copies still needed to reach the next star, given current `stars`
    and `amount` owned. Ported from `calculateAmountToNextLevel`; 0 once a
    card is at the hard 6-star ceiling."""
    if stars >= HARD_STAR_CAP:
        return 0
    s = stars + 1
    req = per_tier * (s + (s // 4 + 16 * (s // 5) + 100 * (s // 6))) ** 2
    return math.ceil(req - amount) + 1


def recommend_cards(
    counts: dict[str, int],
    card_meta: dict[str, dict],
    max_stars: int = BASE_MAX_STARS,
    five_star_set: frozenset[str] | set[str] | None = None,
) -> list[dict]:
    """Owned cards ranked by how FEW copies they need for their next star.

    A card is a candidate iff it's owned (amount > 0), known (in
    `card_meta`), below the account `max_stars` cap, and not in the
    five-star auto-max list. Cheapest next-star first (ties: fewer copies,
    then higher current stars).

    Returns dicts (best first):
        {code, name, amount, stars, to_next, next_total, effect, set}
    """
    five = five_star_set or frozenset()
    out: list[dict] = []
    for code, amount in counts.items():
        if not amount or amount <= 0:
            continue
        meta = card_meta.get(code)
        if meta is None:
            continue  # unknown card code — don't guess thresholds
        if code in five:
            continue  # auto-maxed via the five-star list
        per_tier = meta["per_tier"]
        stars = calculate_stars(per_tier, amount, code, max_stars, code in five)
        if stars >= max_stars:
            continue  # already at the account star cap — no next star
        to_next = copies_to_next_star(per_tier, stars, amount)
        out.append({
            "code": code,
            "name": meta.get("name", code),
            "amount": amount,
            "stars": stars,
            "to_next": to_next,
            "next_total": amount + to_next,
            "effect": meta.get("effect", ""),
            "set": meta.get("set", ""),
        })
    out.sort(key=lambda c: (c["to_next"], -c["stars"]))
    return out
