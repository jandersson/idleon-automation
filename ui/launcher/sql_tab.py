"""SQL tab: a lightweight read-only query browser over the per-bot DBs.

Top row: DB picker + Run button. Middle: query editor pre-seeded with a
useful starting query for the selected DB. Bottom: results table (columns
from the cursor description). Queries run with PRAGMA query_only — no DDL
or DML can run from here.
"""
import sqlite3
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from ui.launcher import config, theme


def build(parent: ttk.Frame, app) -> None:
    # Discover available DBs by scanning minigames/<bot>/assets/*.db
    dbs: dict[str, Path] = {}
    for mg_dir in sorted(config.MINIGAMES_DIR.iterdir()):
        if not mg_dir.is_dir():
            continue
        for db_path in (mg_dir / "assets").glob("*.db"):
            dbs[f"{mg_dir.name} / {db_path.name}"] = db_path
    if not dbs:
        ttk.Label(parent, text="No DBs found under minigames/*/assets/.").pack(pady=20)
        return

    top = ttk.Frame(parent)
    top.pack(fill="x", padx=8, pady=6)
    ttk.Label(top, text="DB:").pack(side="left")
    db_var = tk.StringVar(value=next(iter(dbs)))
    db_combo = ttk.Combobox(top, textvariable=db_var, state="readonly",
                            values=list(dbs.keys()), width=40)
    db_combo.pack(side="left", padx=(4, 12))
    ttk.Button(top, text="Run (Ctrl+Enter)", width=18,
               command=lambda: run_query(dbs[db_var.get()],
                                         query_text.get("1.0", "end").strip(),
                                         tree, status)).pack(side="left")
    status = ttk.Label(top, text="", style="Muted.TLabel")
    status.pack(side="left", padx=(12, 0))

    # Query editor
    query_frame = ttk.LabelFrame(parent, text="Query", padding=4)
    query_frame.pack(fill="x", padx=8, pady=4)
    query_text = tk.Text(query_frame, height=8, wrap="none", font=("Consolas", 10),
                         bg=theme.LOG_BG, fg=theme.TEXT, insertbackground=theme.TEXT,
                         relief="flat", borderwidth=0, padx=6, pady=4)
    query_text.pack(fill="x")
    query_text.bind("<Control-Return>",
                    lambda e: (run_query(dbs[db_var.get()],
                                         query_text.get("1.0", "end").strip(),
                                         tree, status), "break")[1])

    # When DB changes, reset the query to a sensible starter
    def _on_db_change(*_):
        query_text.delete("1.0", "end")
        query_text.insert("1.0", starter_query(dbs[db_var.get()]))
    db_var.trace_add("write", _on_db_change)
    _on_db_change()

    # Results table — Treeview with dynamic columns built per query
    results_frame = ttk.LabelFrame(parent, text="Results", padding=4)
    results_frame.pack(fill="both", expand=True, padx=8, pady=(4, 8))
    tree = ttk.Treeview(results_frame, show="headings")
    ysb = ttk.Scrollbar(results_frame, orient="vertical", command=tree.yview)
    xsb = ttk.Scrollbar(results_frame, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
    ysb.pack(side="right", fill="y")
    xsb.pack(side="bottom", fill="x")
    tree.pack(side="left", fill="both", expand=True)


def starter_query(db_path: Path) -> str:
    """Return a sensible read-only starter query for the given DB.
    Tailored per-bot since the schemas differ; falls back to a
    generic 'show schema'."""
    name = db_path.name
    if name == "mining.db":
        return (
            "-- Survival rate per pit-distance bin\n"
            "SELECT (next_distance_px / 10) * 10 AS dist_bin,\n"
            "       COUNT(*) AS jumps,\n"
            "       SUM(CASE WHEN outcome = 'survived' THEN 1 ELSE 0 END) AS survived,\n"
            "       SUM(CASE WHEN outcome = 'died' THEN 1 ELSE 0 END) AS died\n"
            "FROM jumps\n"
            "WHERE next_kind = 'pit' AND outcome IS NOT NULL\n"
            "GROUP BY dist_bin\n"
            "ORDER BY dist_bin"
        )
    if name == "shots.db":
        return (
            "-- Recent hoops shots\n"
            "SELECT fired_at, hoop_x, hoop_y, \"offset\", made, score_diff, direction\n"
            "FROM shots\n"
            "ORDER BY id DESC LIMIT 50"
        )
    if name == "darts.db":
        return (
            "-- Recent darts throws\n"
            "SELECT fired_at, launch_angle_deg, landing_x, hit, bullseye, streak\n"
            "FROM throws\n"
            "ORDER BY id DESC LIMIT 50"
        )
    return "SELECT name, sql FROM sqlite_master WHERE type IN ('table', 'view')"


def run_query(db_path: Path, sql: str, tree: ttk.Treeview, status: ttk.Label) -> None:
    """Execute the query against the selected DB and load results
    into the tree. Read-only enforced via sqlite3 query_only PRAGMA —
    no DDL or DML can run from this tab."""
    if not sql:
        status.config(text="empty query", foreground=theme.MUTED)
        return
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only = ON")
        cur = conn.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
    except sqlite3.Error as e:
        status.config(text=f"error: {e}", foreground=theme.DANGER)
        return
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass

    # Reset tree
    for c in tree["columns"]:
        tree.heading(c, text="")
    tree.delete(*tree.get_children())
    tree["columns"] = cols
    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, width=max(80, min(300, len(c) * 12)), anchor="w")
    for r in rows:
        tree.insert("", "end", values=tuple("" if v is None else v for v in r))
    status.config(text=f"{len(rows)} row(s)", foreground=theme.SUCCESS)
