"""Stamp upgrade recommender — cheapest material you can afford to level next.

Pure logic, no Tk / no IO. Stamps level by spending an escalating material
cost (and trivial gold), with no save-side cap — so unlike the card/book
recommenders this ranks by *material cost + whether you own the material*,
not by headroom.

Cost formulas are ported from IdleonToolbox `parsers/world-1/stamps.ts`
(`getGoldCost` / `getMaterialCost`). The many reduction sources there (alchemy
vials/sigils, the Atom Collider stamp reducer, the W2 meritocracy bonus, gilded
stamps) are **omitted**: they're all zero/absent early-game, and every one only
*lowers* cost, so the material figure here is an upper bound — a stamp flagged
affordable is genuinely affordable. See #96 to fold reductions in.
"""
from __future__ import annotations

import math

try:
    from ui.launcher.stamp_data import STAMP_META
except Exception:  # not generated yet — degrade to empty
    STAMP_META = {}


def gold_cost(level: int, s: dict) -> int:
    """Gold cost of the next level. `pow_coin` clamps to 1.05, so this stays
    small — gold is rarely the binding constraint."""
    req = s.get("req_mult") or 1
    pow_base = max(1.05, s["pow_coin"] - (level / (level + 5 * req)) * 0.25)
    return math.floor(s["base_coin"] * pow_base ** (level * (10 / req)))


def material_cost(level: int, s: dict) -> int:
    """Material count for the next level (upper bound — reductions omitted)."""
    req = s.get("req_mult") or 1
    step = max(0, round(level / req) - 1)
    return max(1, math.floor(s["base_mat"] * s["pow_mat"] ** (step ** 0.8)))


def recommend_stamps(
    stamp_levels: dict[str, list[int]],
    owned_materials: dict[str, float],
    meta: dict[str, list[dict]] = STAMP_META,
    top_n: int | None = None,
) -> list[dict]:
    """Rank leveled stamps by the material cost of their next level.

    Affordable-first (you own enough of the required material), then cheapest
    material cost. Level-0 stamps are skipped — they may not be owned yet, and
    the list is about cheap next levels for stamps you already have.

    `stamp_levels` is `read_stamps()` shape; `owned_materials` is
    `read_materials()` shape (rawName -> qty). Returns dicts best-first:
        {tab, index, name, level, mat_name, mat_raw, mat_cost, have,
         affordable, gold_cost}
    """
    out: list[dict] = []
    for tab, levels in stamp_levels.items():
        tab_meta = meta.get(tab, [])
        for i, lvl in enumerate(levels):
            if i >= len(tab_meta) or lvl <= 0:
                continue
            s = tab_meta[i]
            mc = material_cost(lvl, s)
            have = owned_materials.get(s.get("mat_raw"), 0)
            out.append({
                "tab": tab,
                "index": i,
                "name": s.get("name", f"{tab}#{i}"),
                "level": lvl,
                "mat_name": s.get("mat_name", "?"),
                "mat_raw": s.get("mat_raw"),
                "mat_cost": mc,
                "have": have,
                "affordable": have >= mc,
                "gold_cost": gold_cost(lvl, s),
            })
    out.sort(key=lambda r: (not r["affordable"], r["mat_cost"]))
    return out[:top_n] if top_n else out
