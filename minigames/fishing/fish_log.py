"""SQLite cast log for the fishing minigame bot.

Fishing's action is a "cast": hold the mouse for `hold_ms` to charge, then
release to land the lure a distance set by that hold. One row per cast
captures the control state at fire time (hold_ms, where the lure was
aimed, the chosen fish) plus the measured outcome a beat later (where it
landed, what it hit, points). That's what fits the hold_ms <-> distance
cast model (cast_model.fit_cast_model) and answers "which hold lands which
fish".

Mirrors minigames/darts/shot_log.py & catching/catch_log.py: a per-action
table (`casts`) + a per-run summary (`runs`), the shared common.db_log
plumbing, an append-only late-column list, open_db / log_* / set_outcome.
DB named for the game (fishing.db), table for the action (casts), per the
CLAUDE.md convention.

Usage:
    from minigames.fishing.fish_log import open_db, log_cast, fetch_cast_samples
    conn = open_db(Path("minigames/fishing/assets/fishing.db"))
    row_id = log_cast(conn, session_started="...", cast_idx=1, hold_ms=420, ...)
    conn.close()
"""
import sqlite3
from pathlib import Path

from common.db_log import open_log_db, insert_row, update_row


_SCHEMA = """
CREATE TABLE IF NOT EXISTS casts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_started TEXT,
    attempt_idx INTEGER,
    cast_idx INTEGER,
    fired_at TEXT,
    hold_ms INTEGER,                -- the control: how long LMB was held
    aim_mode TEXT,                  -- 'model' | 'explore' | 'fallback'
    cast_origin_x INTEGER,          -- lure start x (play-region-relative)
    cast_origin_y INTEGER,
    target_kind TEXT,               -- chosen fish kind, or 'explore'
    target_x INTEGER,               -- chosen fish x at fire
    target_dist_px INTEGER,         -- |target_x - origin_x| the cast aimed for
    predicted_dist_px INTEGER,      -- model's distance_for_hold(hold_ms); NULL when exploring
    landed_x INTEGER,               -- measured lure x after the cast (NULL until lure template exists)
    landed_dist_px INTEGER,         -- |landed_x - origin_x|; the model's training y
    landed_kind TEXT,               -- fish kind under the lure, or NULL/'miss'
    points INTEGER,                 -- 1/2/3/5 on a fish, 0 on a miss, NULL if unmeasured
    made INTEGER,                   -- 1 if landed on any fish, 0 miss, NULL unmeasured
    streak_before INTEGER,          -- consecutive-fish streak at fire
    n_fish INTEGER,                 -- scene complexity at fire
    n_mines INTEGER,
    window_w INTEGER,
    window_h INTEGER,
    outcome TEXT,                   -- 'survived' | 'lost' | NULL (game-over detection unconfirmed)
    outcome_measured_ms INTEGER,
    code_commit TEXT,
    source TEXT
)
"""


_RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_started TEXT,
    attempt_idx INTEGER,
    ended_at TEXT,
    n_casts INTEGER,
    points_total INTEGER,          -- summed points this run (NULL until landing detection)
    max_streak INTEGER,
    duration_s REAL,
    end_reason TEXT,               -- 'process_exit' | 'keyboard_interrupt' | exception name
    final_score INTEGER,           -- NULL until score OCR exists
    code_commit TEXT
)
"""


# Append-only late columns; open_db ALTERs them in on existing DBs.
_LATE_COLUMNS: list[tuple[str, str]] = [
    # Charge-bar fill height at release (px) — the robust cast-power signal
    # (always in-crop, stable), logged even when the bobber lands off-crop.
    # The basis for a charge<->distance model + charge-based aiming (#58).
    ("charge_level", "INTEGER"),
]
_RUNS_LATE_COLUMNS: list[tuple[str, str]] = []


def open_db(path: Path) -> sqlite3.Connection:
    return open_log_db(
        path, [_SCHEMA, _RUNS_SCHEMA],
        {"casts": _LATE_COLUMNS, "runs": _RUNS_LATE_COLUMNS},
    )


def log_cast(conn: sqlite3.Connection, **fields) -> int:
    """Insert a cast row; return its rowid so set_outcome can backfill the
    landing once the lure is located post-cast."""
    return insert_row(conn, "casts", fields)


def set_outcome(conn: sqlite3.Connection, row_id: int, **fields) -> None:
    """Backfill measured landing/outcome on a cast row (landed_x,
    landed_dist_px, landed_kind, points, made, outcome, ...)."""
    update_row(conn, "casts", row_id, fields)


def log_run(conn: sqlite3.Connection, **fields) -> int:
    """Insert a per-run summary row at process exit. Returns the rowid."""
    return insert_row(conn, "runs", fields)


def fetch_cast_samples(conn: sqlite3.Connection, source: str | None = None) -> list[tuple[int, float]]:
    """(hold_ms, landed_dist_px) pairs for the cast-model fit. Only rows
    with both measured are returned. `source=None` uses every row; pass
    'bot' to exclude any future human-watch casts."""
    where = ["hold_ms IS NOT NULL", "landed_dist_px IS NOT NULL"]
    params: list = []
    if source is not None:
        where.append("source = ?")
        params.append(source)
    rows = conn.execute(
        f"SELECT hold_ms, landed_dist_px FROM casts WHERE {' AND '.join(where)}",
        params,
    ).fetchall()
    return [(int(h), float(d)) for h, d in rows]
