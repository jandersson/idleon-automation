"""Stamps tab: cheapest stamp upgrades you can afford to level next.

Stamps level by material cost (gold is trivial — see stamp_recommender), so
this ranks leveled stamps by next-level material cost and whether you own the
material: the actionable "quick wins" list, affordable ones first. Cost data is
vendored (stamp_data); reduction sources aren't modelled yet (#96), so material
costs are a safe upper bound (a stamp shown affordable genuinely is).
"""
import tkinter as tk
from tkinter import ttk

from common.idleon_save import read_stamps, read_materials
from ui.launcher import scroll, theme
from ui.launcher.stamp_recommender import recommend_stamps

TOP_N = 40
_TAB_LABEL = {"combat": "Combat", "skills": "Skills", "misc": "Misc"}


def _fmt(n: float) -> str:
    """Compact large counts: 1234567 -> 1.23M."""
    n = float(n)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= div:
            return f"{n / div:.2f}{suf}"
    return str(int(n))


def build(parent: ttk.Frame, app) -> None:
    controls = ttk.Frame(parent, padding=6)
    controls.pack(fill="x")
    ttk.Label(controls, text="Cheapest stamp upgrades you can afford next").pack(side="left")
    ttk.Button(controls, text="Refresh", command=lambda: refresh(app)).pack(side="left", padx=(12, 0))
    app.stamps_status = ttk.Label(controls, text="", style="Muted.TLabel")
    app.stamps_status.pack(side="left", padx=12)
    ttk.Label(controls, text="gold cost is negligible — material is the gate",
              style="Muted.TLabel").pack(side="right")

    body = ttk.Frame(parent)
    body.pack(fill="both", expand=True, padx=6, pady=(0, 6))
    app.stamps_canvas = tk.Canvas(body, bg=theme.BG, highlightthickness=0)
    app.stamps_canvas.pack(side="left", fill="both", expand=True)
    sb = ttk.Scrollbar(body, orient="vertical", command=app.stamps_canvas.yview)
    sb.pack(side="right", fill="y")
    app.stamps_canvas.config(yscrollcommand=sb.set)
    app.stamps_inner = ttk.Frame(app.stamps_canvas)
    app.stamps_canvas.create_window((0, 0), window=app.stamps_inner, anchor="nw")
    app.stamps_inner.bind(
        "<Configure>",
        lambda _e: app.stamps_canvas.configure(scrollregion=app.stamps_canvas.bbox("all")),
    )
    scroll.bind_mousewheel(app.stamps_canvas)

    refresh(app)


def refresh(app) -> None:
    for child in app.stamps_inner.winfo_children():
        child.destroy()

    stamps = read_stamps()
    if stamps is None:
        app.stamps_status.config(
            text="couldn't read save (Idleon not installed, or plyvel missing)")
        return
    materials = read_materials() or {}
    recs = recommend_stamps(stamps, materials)
    if not recs:
        ttk.Label(app.stamps_inner, text="no leveled stamps found in the save",
                  style="Muted.TLabel").pack(anchor="w", padx=8, pady=8)
        app.stamps_status.config(text="0 leveled stamps")
        return

    affordable = sum(1 for r in recs if r["affordable"])
    for r in recs[:TOP_N]:
        row = ttk.Frame(app.stamps_inner)
        row.pack(fill="x", padx=4, pady=1)
        color = theme.SUCCESS if r["affordable"] else theme.MUTED
        mark = "✓" if r["affordable"] else "·"
        ttk.Label(row, text=f"{mark} {r['name']}", width=28,
                  foreground=color).pack(side="left")
        ttk.Label(row, text=_TAB_LABEL.get(r["tab"], r["tab"]), width=8,
                  style="Muted.TLabel").pack(side="left")
        ttk.Label(row, text=f"Lv {r['level']}", width=7,
                  style="Muted.TLabel").pack(side="left")
        ttk.Label(row, text=f"{_fmt(r['mat_cost'])} {r['mat_name']}", width=22,
                  foreground=theme.INFO).pack(side="left")
        ttk.Label(row, text=f"have {_fmt(r['have'])}", width=14,
                  foreground=color).pack(side="left")

    app.stamps_status.config(
        text=f"{affordable} affordable now · {len(recs)} leveled stamps"
             + (f" · showing top {TOP_N}" if len(recs) > TOP_N else ""))
