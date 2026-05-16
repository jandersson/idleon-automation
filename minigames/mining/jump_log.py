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
    conn = open_db(Path("minigames/mining/assets/jumps.db"))
    row_id = log_jump(conn, session_started="...", jump_idx=1, ...)
    set_outcome(conn, row_id, "survived", measured_ms=1500)
"""
import sqlite3
from pathlib import Path


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


def _migrate(conn: sqlite3.Connection) -> None:
    for name, decl in _LATE_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE jumps ADD COLUMN {name} {decl}")
        except sqlite3.OperationalError:
            pass


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute(_SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


def log_jump(conn: sqlite3.Connection, **fields) -> int:
    """Insert a jump row, return its rowid so callers can set_outcome
    on it later."""
    cols = ", ".join(fields)
    placeholders = ", ".join("?" * len(fields))
    cur = conn.execute(
        f"INSERT INTO jumps ({cols}) VALUES ({placeholders})",
        tuple(fields.values()),
    )
    conn.commit()
    return cur.lastrowid


def set_outcome(conn: sqlite3.Connection, row_id: int, outcome: str,
                measured_ms: int) -> None:
    """Fill in outcome + outcome_measured_ms on an existing jump row."""
    conn.execute(
        "UPDATE jumps SET outcome = ?, outcome_measured_ms = ? WHERE id = ?",
        (outcome, measured_ms, row_id),
    )
    conn.commit()


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
