"""SQLite cast log for the fishing minigame bot.

Fishing's action is a "cast": hold the mouse to fill the charge bar, then
release to land the lure a distance set by the charge LEVEL. The bot casts
closed-loop (release at a target fill, common.input.charge_and_release), so the
control is `charge_level` — `hold_ms` is retained NULL for legacy/open-loop
compatibility. One row per cast captures the control state at fire time
(target_charge_level, where the lure was aimed, the chosen fish) plus the
measured outcome a beat later (the release charge_level, where it landed, what
it hit, points). That's what fits the charge_level <-> distance cast model
(cast_model.fit_cast_model) and answers "which charge lands which fish".

Mirrors minigames/darts/shot_log.py & catching/catch_log.py: a per-action
table (`casts`) + a per-run summary (`runs`), the shared common.db_log
plumbing, an append-only late-column list, open_db / log_* / set_outcome.
DB named for the game (fishing.db), table for the action (casts), per the
CLAUDE.md convention.

Usage:
    from minigames.fishing.fish_log import open_db, log_cast, fetch_cast_samples
    conn = open_db(Path("minigames/fishing/assets/fishing.db"))
    row_id = log_cast(conn, session_started="...", cast_idx=1,
                      target_charge_level=44, ...)   # set_outcome backfills charge_level
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
    hold_ms INTEGER,                -- legacy open-loop hold; NULL under closed-loop (charge is the control)
    aim_mode TEXT,                  -- 'model' | 'explore' | 'fallback'
    cast_origin_x INTEGER,          -- lure start x (play-region-relative)
    cast_origin_y INTEGER,
    target_kind TEXT,               -- chosen fish kind, or 'explore'
    target_x INTEGER,               -- chosen fish x at fire
    target_dist_px INTEGER,         -- |target_x - origin_x| the cast aimed for
    predicted_dist_px INTEGER,      -- model's distance_for_charge(target_charge); NULL when no model yet
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
    # Charge-bar fill height at release (px) — the cast-power signal and the
    # charge<->distance model's training feature (in-crop, stable). With the
    # closed-loop cast this is the ACTUAL fill the lure released at (#58).
    ("charge_level", "INTEGER"),
    # The charge fill the closed-loop cast aimed to release at (charge_level is
    # what it actually got). Aim error = charge_level - target_charge_level.
    ("target_charge_level", "INTEGER"),
]
# Rod-not-ready aborts this run: closed-loop casts skipped because the bar
# stayed 0 (previous lure still reeling in) — the cast-too-soon waste, now
# detected + retried instead of burned as a dead cast (#58).
_RUNS_LATE_COLUMNS: list[tuple[str, str]] = [
    ("n_not_ready", "INTEGER"),
]


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
    """(charge_level, landed_dist_px) pairs for the cast-model fit. Only rows
    with both measured are returned — charge_level is the closed-loop release
    fill (the clean predictor), landed_dist_px the bobber landing (the target).
    `source=None` uses every row; pass 'bot' to exclude any future human-watch
    casts. (Pre-charge rows have charge_level NULL and are excluded, so the
    old hold-era landings don't pollute the charge fit.)"""
    where = ["charge_level IS NOT NULL", "landed_dist_px IS NOT NULL"]
    params: list = []
    if source is not None:
        where.append("source = ?")
        params.append(source)
    rows = conn.execute(
        f"SELECT charge_level, landed_dist_px FROM casts WHERE {' AND '.join(where)}",
        params,
    ).fetchall()
    return [(int(c), float(d)) for c, d in rows]
