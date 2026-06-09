"""SQLite jump log for the mining bot.

One row per click decision: detector state at fire time + outcome
measured a couple of seconds later (did the cart survive, or did it
fall in / did the attempt end). Mirrors the shape of darts.shot_log.

Each session_started/attempt_idx pair groups jumps within one
"Play Game" attempt. attempt_idx increments every time the bot
clicks the Play Game button; jump_idx is per-attempt.

Outcomes (filled in after a settle delay):
    "survived" — cart is still detected on the plank, attempt continues
    "died"     — plank gone (game-over screen back), attempt ended
    "unknown"  — neither: cart momentarily not detected but plank still
                 there, or measurement window expired without a clear signal

Usage:
    from minigames.mining.jump_log import open_db, log_jump, set_outcome
    conn = open_db(Path("minigames/mining/assets/mining.db"))
    row_id = log_jump(conn, session_started="...", jump_idx=1, ...)
    set_outcome(conn, row_id, "survived", measured_ms=1500)
"""
import sqlite3
from pathlib import Path

from common.db_log import open_log_db, insert_row, update_row


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jumps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_started TEXT,
    attempt_idx INTEGER,
    jump_idx INTEGER,
    clicked_at TEXT,
    cart_x INTEGER,
    cart_y INTEGER,
    next_kind TEXT,
    next_x INTEGER,
    next_distance_px INTEGER,
    plank_y INTEGER,
    plank_x_left INTEGER,
    plank_x_right INTEGER,
    window_w INTEGER,
    window_h INTEGER,
    outcome TEXT,
    outcome_measured_ms INTEGER,
    code_commit TEXT,
    source TEXT
)
"""


_LATE_COLUMNS: list[tuple[str, str]] = [
    # (no late columns yet — placeholder for future additions)
]


def open_db(path: Path) -> sqlite3.Connection:
    return open_log_db(path, [_SCHEMA], {"jumps": _LATE_COLUMNS})


def log_jump(conn: sqlite3.Connection, **fields) -> int:
    """Insert a jump row, return its rowid so callers can set_outcome
    on it later."""
    return insert_row(conn, "jumps", fields)


def set_outcome(conn: sqlite3.Connection, row_id: int, outcome: str,
                measured_ms: int) -> None:
    """Fill in outcome + outcome_measured_ms on an existing jump row."""
    update_row(conn, "jumps", row_id,
               {"outcome": outcome, "outcome_measured_ms": measured_ms})


def survival_rate_by_distance(conn: sqlite3.Connection,
                              kind: str = "pit") -> list[tuple[int, int, int]]:
    """Return [(distance_bin, n_total, n_survived), ...] for jumps of
    the given kind, grouped into 10-px distance bins. Quick way to see
    'jumps fired at pit_dist=50 survived 8/10, pit_dist=70 survived
    2/10 — tighten the trigger window.'"""
    rows = conn.execute(
        '''
        SELECT (next_distance_px / 10) * 10 AS bin,
               COUNT(*),
               SUM(CASE WHEN outcome = 'survived' THEN 1 ELSE 0 END)
        FROM jumps
        WHERE next_kind = ? AND next_distance_px IS NOT NULL
        GROUP BY bin
        ORDER BY bin
        ''',
        (kind,),
    ).fetchall()
    return [(int(b), int(n), int(s or 0)) for b, n, s in rows]
