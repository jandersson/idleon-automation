"""Books tab (#44): per-character Library talent-book recommendations.

Reads each character's talent levels + caps from the save and lists the
talents that are AT CAP and would benefit from checking out a higher book,
ranked by importance and cap-gap. The talent name/importance/exclusion
metadata comes from ui/launcher/talent_data.py (sourced separately); if it
isn't present yet the tab still works, naming talents by index.

Max book level isn't a single save field (it's summed from merit/salt/
achievement/atom/sailing/summoning), so it's a user-entered number here,
defaulting to the current account ceiling.
"""
import tkinter as tk
from tkinter import ttk

from common.idleon_save import read_talents
from ui.launcher import theme
from ui.launcher.book_recommender import recommend_books

try:
    from ui.launcher.talent_data import TALENT_META
except Exception:  # not sourced yet / malformed — degrade to index names
    TALENT_META = {}

DEFAULT_MAX_BOOK_LEVEL = 396
TOP_N_PER_CHARACTER = 8


def build(parent: ttk.Frame, app) -> None:
    controls = ttk.Frame(parent, padding=6)
    controls.pack(fill="x")
    ttk.Label(controls, text="Max book level:").pack(side="left")
    app.max_book_var = tk.StringVar(value=str(DEFAULT_MAX_BOOK_LEVEL))
    ttk.Entry(controls, textvariable=app.max_book_var, width=6).pack(side="left", padx=(4, 12))
    ttk.Button(controls, text="Refresh", command=lambda: refresh(app)).pack(side="left")
    app.books_status = ttk.Label(controls, text="", style="Muted.TLabel")
    app.books_status.pack(side="left", padx=12)
    if not TALENT_META:
        ttk.Label(controls, text="(talent names not sourced yet — showing indices)",
                  style="Muted.TLabel").pack(side="left", padx=12)

    # Scrollable results body (a character can have several recommendations).
    body = ttk.Frame(parent)
    body.pack(fill="both", expand=True, padx=6, pady=(0, 6))
    app.books_canvas = tk.Canvas(body, bg=theme.BG, highlightthickness=0)
    app.books_canvas.pack(side="left", fill="both", expand=True)
    sb = ttk.Scrollbar(body, orient="vertical", command=app.books_canvas.yview)
    sb.pack(side="right", fill="y")
    app.books_canvas.config(yscrollcommand=sb.set)
    app.books_inner = ttk.Frame(app.books_canvas)
    app.books_canvas.create_window((0, 0), window=app.books_inner, anchor="nw")
    app.books_inner.bind(
        "<Configure>",
        lambda _e: app.books_canvas.configure(scrollregion=app.books_canvas.bbox("all")),
    )

    refresh(app)


def refresh(app) -> None:
    for child in app.books_inner.winfo_children():
        child.destroy()
    try:
        max_book = int(app.max_book_var.get())
    except ValueError:
        app.books_status.config(text="max book level must be an integer")
        return

    talents = read_talents()
    if talents is None:
        app.books_status.config(text="couldn't read save (Idleon not installed, or plyvel missing)")
        return
    if not talents:
        app.books_status.config(text="no characters with talent data in the save")
        return

    total = 0
    for name, t in talents.items():
        recs = recommend_books(
            t["skill_levels"], t["skill_levels_max"], TALENT_META, max_book,
        )
        card = ttk.LabelFrame(app.books_inner, text=_char_header(name, t, len(recs)),
                              padding=6)
        card.pack(fill="x", padx=4, pady=4)
        if not recs:
            ttk.Label(card, text="nothing at cap below the max book level",
                      style="Muted.TLabel").pack(anchor="w")
            continue
        for r in recs[:TOP_N_PER_CHARACTER]:
            total += 1
            row = ttk.Frame(card)
            row.pack(fill="x")
            tag = {1: "★", 2: "▲", 3: "•", 4: "·"}.get(r["importance"], "•")
            ttk.Label(row, text=f"{tag} {r['name']}", width=26).pack(side="left")
            ttk.Label(row, text=f"cap {r['cap']} → up to {max_book}  (+{r['gap']})",
                      foreground=theme.INFO).pack(side="left")

    app.books_status.config(
        text=f"{total} book candidate(s) across {len(talents)} character(s)")


def _char_header(name: str, t: dict, n_recs: int) -> str:
    cls = t.get("character_class")
    cls_str = f" (class {cls})" if cls is not None else ""
    return f"{name}{cls_str} — {n_recs} candidate(s)"
