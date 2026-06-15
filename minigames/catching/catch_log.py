"""SQLite flap log for the catching (Flappy-style) bot.

Catching's action is a "flap": a click that lifts the falling fly. One row
per flap captures the control state at fire time — the fly's position and
the next ring-gap it's threading — so an observe/analysis pass can answer
"at what fly-vs-gap offset does the bot click, and (once game-over
detection exists) which offsets keep the fly alive?".

Mirrors minigames/mining/jump_log.py and minigames/darts/shot_log.py: a
per-action table + a per-run summary, the shared common.db_log plumbing, an
append-only late-column list, and open_db / log_* / set_outcome helpers. The
DB is named for the game (catching.db), the table for the action (flaps),
per the CLAUDE.md convention.

Outcome is NULL for now: the bot has no game-over / attempt detection yet
(issue #47), so the fatal-flap label can't be filled until that lands. The
column and set_outcome are scaffolded so adding it is a migration-free
change.

Usage:
    from minigames.catching.catch_log import open_db, log_flap
    conn = open_db(Path("minigames/catching/assets/catching.db"))
    row_id = log_flap(conn, session_started="...", flap_idx=1, ...)
    conn.close()
"""
import sqlite3
from pathlib import Path

from common.db_log import open_log_db, insert_row, update_row


_SCHEMA = """
CREATE TABLE IF NOT EXISTS flaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_started TEXT,
    attempt_idx INTEGER,
    flap_idx INTEGER,
    clicked_at TEXT,
    fly_x INTEGER,
    fly_y INTEGER,
    fly_vy INTEGER,                 -- fly vertical velocity at fire, px/SECOND (+ = descending); NULL when no recent prior frame
    gap_top INTEGER,
    gap_bottom INTEGER,
    gap_left_x INTEGER,             -- next ring's horizontal extent (how far ahead it is)
    gap_right_x INTEGER,
    gap_center INTEGER,
    gap_height INTEGER,
    fly_offset_below_gap INTEGER,   -- fly_y - gap_bottom at fire (the decision quantity)
    gap_lower_margin INTEGER,       -- GAP_LOWER_MARGIN policy constant at fire
    window_w INTEGER,
    window_h INTEGER,
    outcome TEXT,                   -- 'survived' | 'died' | NULL (no game-over detection yet, #47)
    outcome_measured_ms INTEGER,
    code_commit TEXT,
    source TEXT
)
"""


# Per-run summary. One row per bot process today (no in-game attempt
# detection yet — see #47), written at exit with how the run ended and how
# many flaps it fired. final_score stays NULL until score OCR exists.
_RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_started TEXT,
    attempt_idx INTEGER,
    ended_at TEXT,
    n_flaps INTEGER,
    duration_s REAL,
    end_reason TEXT,                -- 'process_exit' | 'keyboard_interrupt' | exception name
    final_score INTEGER,           -- NULL until score OCR exists
    code_commit TEXT
)
"""


# New columns added after the original schema; open_db ALTERs them in on
# existing DBs. Keep append-only — same pattern as the other bots' logs.
_LATE_COLUMNS: list[tuple[str, str]] = []
_RUNS_LATE_COLUMNS: list[tuple[str, str]] = []


def open_db(path: Path) -> sqlite3.Connection:
    return open_log_db(
        path, [_SCHEMA, _RUNS_SCHEMA],
        {"flaps": _LATE_COLUMNS, "runs": _RUNS_LATE_COLUMNS},
    )


def log_flap(conn: sqlite3.Connection, **fields) -> int:
    """Insert a flap row; return its rowid so set_outcome can backfill the
    fatal-flap label once game-over detection exists."""
    return insert_row(conn, "flaps", fields)


def set_outcome(conn: sqlite3.Connection, row_id: int, outcome: str,
                measured_ms: int) -> None:
    """Fill in outcome + outcome_measured_ms on an existing flap row."""
    update_row(conn, "flaps", row_id,
               {"outcome": outcome, "outcome_measured_ms": measured_ms})


def log_run(conn: sqlite3.Connection, **fields) -> int:
    """Insert a per-run summary row at process exit. Returns the rowid."""
    return insert_row(conn, "runs", fields)


def offset_histogram(
    conn: sqlite3.Connection,
    source: str | None = "bot",
    bin_px: int = 5,
) -> list[tuple[int, int]]:
    """[(offset_bin, count), ...]: distribution of fly_offset_below_gap
    (fly_y - gap_bottom at fire) over `bin_px`-wide bins. The first thing
    to look at on an observe run — it shows WHERE relative to the gap the
    bot decides to flap, before any outcome labels exist.

    source defaults to 'bot' so the bot's policy isn't blended with human
    watch clicks; pass None for all rows or 'human' for watch data. Bins
    truncate toward zero (SQLite integer division), so the bin straddling
    zero is slightly wider than bin_px — fine for a rough histogram.

    Once enough rows exist, slice this by fly_vy (descent speed at fire):
    the offset that's safe at a slow descent may not be at a fast one, the
    same way darts found the outcome rides on vy, not the trigger alone.
    """
    where = ["fly_offset_below_gap IS NOT NULL"]
    params: list = []
    if source is not None:
        where.append("source = ?")
        params.append(source)
    rows = conn.execute(
        f"""
        SELECT (fly_offset_below_gap / {int(bin_px)}) * {int(bin_px)} AS bin,
               COUNT(*)
        FROM flaps
        WHERE {" AND ".join(where)}
        GROUP BY bin
        ORDER BY bin
        """,
        params,
    ).fetchall()
    return [(int(b), int(n)) for b, n in rows]
