"""Cards tab: which cards are closest to their next star.

Account-wide (cards are shared), so one ranked list: owned cards sorted by
how few more copies they need for the next star — the high-ROI farming
targets. Star math is ported from IdleonToolbox (see card_recommender);
metadata comes from the vendored card_data.CARD_META. Reads everything
from the local save.
"""
import tkinter as tk
from tkinter import ttk

from common.idleon_save import (
    read_card_counts,
    read_card_five_star_set,
    read_card_star_cap,
)
from ui.launcher import scroll, theme
from ui.launcher.card_recommender import recommend_cards

try:
    from ui.launcher.card_data import CARD_META
except Exception:  # not generated yet — degrade gracefully
    CARD_META = {}

TOP_N = 30


def build(parent: ttk.Frame, app) -> None:
    controls = ttk.Frame(parent, padding=6)
    controls.pack(fill="x")
    ttk.Label(controls, text="Cards closest to their next star").pack(side="left")
    ttk.Button(controls, text="Refresh", command=lambda: refresh(app)).pack(side="left", padx=(12, 0))
    app.cards_status = ttk.Label(controls, text="", style="Muted.TLabel")
    app.cards_status.pack(side="left", padx=12)

    body = ttk.Frame(parent)
    body.pack(fill="both", expand=True, padx=6, pady=(0, 6))
    app.cards_canvas = tk.Canvas(body, bg=theme.BG, highlightthickness=0)
    app.cards_canvas.pack(side="left", fill="both", expand=True)
    sb = ttk.Scrollbar(body, orient="vertical", command=app.cards_canvas.yview)
    sb.pack(side="right", fill="y")
    app.cards_canvas.config(yscrollcommand=sb.set)
    app.cards_inner = ttk.Frame(app.cards_canvas)
    app.cards_canvas.create_window((0, 0), window=app.cards_inner, anchor="nw")
    app.cards_inner.bind(
        "<Configure>",
        lambda _e: app.cards_canvas.configure(scrollregion=app.cards_canvas.bbox("all")),
    )
    scroll.bind_mousewheel(app.cards_canvas)

    refresh(app)


_STAR = "★"  # ★


def refresh(app) -> None:
    for child in app.cards_inner.winfo_children():
        child.destroy()

    counts = read_card_counts()
    if counts is None:
        app.cards_status.config(text="couldn't read save (Idleon not installed, or plyvel missing)")
        return
    if not CARD_META:
        app.cards_status.config(text="card data not generated yet (ui/launcher/card_data.py)")
        return

    cap = read_card_star_cap()
    five = read_card_five_star_set()
    recs = recommend_cards(counts, CARD_META, max_stars=cap, five_star_set=five)

    header = ttk.LabelFrame(
        app.cards_inner,
        text=f"Closest to next star  (star cap {cap}{_STAR}, {len(recs)} cards below cap)",
        padding=6,
    )
    header.pack(fill="x", padx=4, pady=4)
    if not recs:
        ttk.Label(header, text="every owned card is at the star cap — nothing to farm toward",
                  style="Muted.TLabel").pack(anchor="w")
    else:
        for r in recs[:TOP_N]:
            row = ttk.Frame(header)
            row.pack(fill="x")
            stars_txt = f"{r['stars']}{_STAR}→{r['stars'] + 1}{_STAR}"
            ttk.Label(row, text=f"{r['name']}", width=22).pack(side="left")
            ttk.Label(row, text=stars_txt, width=8, foreground=theme.ACCENT).pack(side="left")
            ttk.Label(row, text=f"+{r['to_next']} copies", width=12,
                      foreground=theme.INFO).pack(side="left")
            ttk.Label(row, text=f"have {r['amount']}", width=12,
                      style="Muted.TLabel").pack(side="left")
            ttk.Label(row, text=f"{r['effect']}", style="Muted.TLabel").pack(side="left")

    owned = sum(1 for v in counts.values() if v > 0)
    app.cards_status.config(
        text=f"{owned} cards owned; {len(recs)} below the {cap}-star cap")
