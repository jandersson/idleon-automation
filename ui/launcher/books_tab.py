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

from common.idleon_save import read_talents, read_book_checkouts
from ui.launcher import scroll, theme
from ui.launcher.book_recommender import (
    recommend_books, recommend_books_account, special_talents_account,
)

try:
    from ui.launcher.talent_data import TALENT_META, class_name, base_genre
except Exception:  # not sourced yet / malformed — degrade to index names
    TALENT_META = {}

    def class_name(class_id):  # type: ignore[misc]
        return "?" if class_id is None else f"Class {class_id}"

    def base_genre(class_line):  # type: ignore[misc]
        return "?" if class_line is None else class_line

# Max book level is wiki-confirmed (Talent Book Library): the ceiling is
# the sum of merit/salt/achievement/atom/sailing-relic/summoning sources,
# which currently tops out at 396. Not a single save field, so it's a
# user-editable entry defaulting to that ceiling.
DEFAULT_MAX_BOOK_LEVEL = 396
TOP_N_PER_CHARACTER = 8
TOP_N_ACCOUNT = 12  # account-wide "spend the shared pool here next" list
TOP_N_SPECIAL = 12  # star/VIP-bookshelf "nab if you can" list


def build(parent: ttk.Frame, app) -> None:
    controls = ttk.Frame(parent, padding=6)
    controls.pack(fill="x")
    ttk.Label(controls, text="Max book level:").pack(side="left")
    app.max_book_var = tk.StringVar(value=str(DEFAULT_MAX_BOOK_LEVEL))
    ttk.Entry(controls, textvariable=app.max_book_var, width=6).pack(side="left", padx=(4, 12))
    ttk.Button(controls, text="Refresh", command=lambda: refresh(app)).pack(side="left")
    # Shared checkout pool (OLA[55]) — books are spent from one account-wide
    # pool, so this is the budget the account-wide ranking spends down.
    app.books_checkouts = ttk.Label(controls, text="", font=("Segoe UI", 10, "bold"),
                                    foreground=theme.ACCENT)
    app.books_checkouts.pack(side="left", padx=(16, 0))
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
    scroll.bind_mousewheel(app.books_canvas)

    refresh(app)


_IMPORTANCE_TAG = {1: "★", 2: "▲", 3: "•", 4: "·"}


def _book_path(r: dict) -> str:
    """The Library navigation path to a recommended book: genre › subgenre.

    The in-game Talent Book Library is browsed base-class (genre) → subclass
    (subgenre), so this is where to go to find the book. `recommend_books`
    carries the talent's class line as `klass` (the subgenre); its base class
    is the genre. Collapses to a single label for base-class / Beginner /
    Special talents, where genre and subgenre coincide."""
    subgenre = r.get("klass") or "?"
    genre = base_genre(subgenre)
    if genre == subgenre:
        return f"📖 {genre}"
    return f"📖 {genre} › {subgenre}"


def _is_special(r: dict) -> bool:
    """Whether a class-Library candidate is actually a star/special talent.

    A guard for the regular checkout lists: any mapped talent whose genre is
    "Special" (the named-but-mostly-inactive 15-44 block) belongs to the VIP
    Bookshelf, not a class checkout, so it's filtered out of the targetable
    lists. The ACTIVE star block (save slots 600+) is unmapped and handled
    separately via special_talents_account()."""
    return base_genre(r.get("klass")) == "Special"


def refresh(app) -> None:
    for child in app.books_inner.winfo_children():
        child.destroy()
    try:
        max_book = int(app.max_book_var.get())
    except ValueError:
        app.books_status.config(text="max book level must be an integer")
        return

    checkouts = read_book_checkouts()
    app.books_checkouts.config(
        text=f"📚 {checkouts} checkouts available" if checkouts is not None else "")

    talents = read_talents()
    if talents is None:
        app.books_status.config(text="couldn't read save (Idleon not installed, or plyvel missing)")
        return
    if not talents:
        app.books_status.config(text="no characters with talent data in the save")
        return

    # Class-library checkouts (targetable). _is_special guards against the
    # mapped 15-44 "Special" block sneaking in — the ACTIVE star talents live
    # in the unmapped 600+ block, read separately below.
    account = recommend_books_account(talents, TALENT_META, max_book)
    regular = [r for r in account if not _is_special(r)]
    special = special_talents_account(talents)

    # Account-wide section first: checkouts are a shared pool, so this is
    # the "spend your next books here" list across every character.
    acct_card = ttk.LabelFrame(
        app.books_inner,
        text=f"Account-wide — best books to check out next ({len(regular)} total)",
        padding=6,
    )
    acct_card.pack(fill="x", padx=4, pady=4)
    if not regular:
        ttk.Label(acct_card, text="nothing at cap below the max book level",
                  style="Muted.TLabel").pack(anchor="w")
    else:
        for r in regular[:TOP_N_ACCOUNT]:
            row = ttk.Frame(acct_card)
            row.pack(fill="x")
            tag = _IMPORTANCE_TAG.get(r["importance"], "•")
            ttk.Label(row, text=f"{tag} {r['name']}", width=26).pack(side="left")
            ttk.Label(row, text=_book_path(r), width=27,
                      foreground=theme.ACCENT).pack(side="left")
            ttk.Label(row, text=f"{r['character']}", width=14,
                      style="Muted.TLabel").pack(side="left")
            ttk.Label(row, text=f"cap {r['cap']} → up to {max_book}  (+{r['gap']})",
                      foreground=theme.INFO).pack(side="left")

    # Star/special talents at cap — not a targetable checkout, so their own
    # section. Read straight from the save's 600+ block; unmapped, so shown by
    # slot, not name (this block has no authoritative index→name source).
    if special:
        sp_card = ttk.LabelFrame(
            app.books_inner,
            text=f"Special / Star talents at cap — nab if you can ({len(special)})",
            padding=6,
        )
        sp_card.pack(fill="x", padx=4, pady=4)
        ttk.Label(
            sp_card,
            text="These are at cap on at least one character — a higher Star "
                 "Talent Book would raise them. Not a targetable checkout: nab a "
                 "random one at the VIP Bookshelf (5 checkouts) or from Special "
                 "Talent Book drops/quests. Shown by save slot (block isn't named).",
            style="Muted.TLabel", wraplength=620, justify="left",
        ).pack(anchor="w", pady=(0, 4))
        for r in special[:TOP_N_SPECIAL]:
            row = ttk.Frame(sp_card)
            row.pack(fill="x")
            ttk.Label(row, text=f"🌟 special talent · slot {r['index']}",
                      width=28).pack(side="left")
            ttk.Label(row, text=f"cap {r['cap']}  ·  at cap",
                      foreground=theme.WARN, width=18).pack(side="left")
            ttk.Label(row, text=f"on {r['character']}", width=16,
                      style="Muted.TLabel").pack(side="left")

    # Per-character breakdown below (class-library checkouts only — the star
    # talents above are shared across every character).
    total = 0
    for name, t in talents.items():
        recs = [r for r in recommend_books(
            t["skill_levels"], t["skill_levels_max"], TALENT_META, max_book,
        ) if not _is_special(r)]
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
            tag = _IMPORTANCE_TAG.get(r["importance"], "•")
            ttk.Label(row, text=f"{tag} {r['name']}", width=26).pack(side="left")
            ttk.Label(row, text=_book_path(r), width=27,
                      foreground=theme.ACCENT).pack(side="left")
            ttk.Label(row, text=f"cap {r['cap']} → up to {max_book}  (+{r['gap']})",
                      foreground=theme.INFO).pack(side="left")

    app.books_status.config(
        text=f"{total} book candidate(s) across {len(talents)} character(s)"
             + (f" · {len(special)} star talent(s) to nab" if special else ""))


def _char_header(name: str, t: dict, n_recs: int) -> str:
    cls = class_name(t.get("character_class"))
    return f"{name} ({cls}) — {n_recs} candidate(s)"
