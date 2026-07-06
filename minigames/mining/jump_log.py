"""SQLite jump log for the mining bot.

One row per click decision: detector state at fire time + outcome
measured a couple of seconds later (did the cart survive, or did it
fall in / did the attempt end). Mirrors the shape of darts.shot_log.

Each session_started/attempt_idx pair groups jumps within one
"Play Game" attempt. jump_idx is per-attempt. attempt_idx is a placeholder
that is always 1 today: the bot runs one attempt per process, so a fresh
process (new session_started) bounds each attempt. (A future multi-attempt
replay loop would increment it.)

Outcomes (filled in after a settle delay):
    "survived" — cart still detected (grounded) a beat later, attempt continues
    "died"     — cart gone (game-over), attempt ended
    "unknown"  — LEGACY only. The current settle code writes just
                 survived/died (the old plank_y branch produced 'unknown'
                 when a death left the plank briefly visible). survival
                 queries filter to survived/died.

next_kind is 'pit' or 'ore' (NULL when no obstacle was in view). Ore
detection went live 2026-06-16 (above-plank brown signature, see
detector._scan_plank_ore); rows before that are pit/NULL only.

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
    # Phase-D (slam scoring) hooks. Added ahead of the first scoring run so it
    # records slam ground truth without a mid-collection schema bump. Migrated
    # idempotently via open_log_db's ALTER list; old rows backfill NULL.
    ("action", "TEXT"),                    # 'jump' | 'slam'  (NULL == legacy jump)
    ("cart_height_above_plank", "INTEGER"),  # plank_y - cart_y at click (altitude)
    ("cart_pose", "TEXT"),                 # matched cart template name
    ("score_at_click", "INTEGER"),         # last OCR'd PTS at fire time
    # Speed-adaptive triggers (#53). next_width_px is the obstacle's x-extent
    # (the gap width W of the Run-11 feasibility bound); scroll_v_px_s is the
    # live ScrollVelocity estimate at fire time (NULL = estimator not warmed
    # up, i.e. the static windows were in effect).
    ("next_width_px", "INTEGER"),
    ("scroll_v_px_s", "REAL"),
]


# Per-attempt run summary (#6). One row per "Play Game" attempt, written
# when the run ends — captures the final PTS score and how the run ended.
_RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_started TEXT,
    attempt_idx INTEGER,
    ended_at TEXT,
    final_score INTEGER,
    end_reason TEXT,        -- 'play_button' (prompt returned), 'plank_lost' (timeout), 'failsafe'
    code_commit TEXT
)
"""

_RUNS_LATE_COLUMNS: list[tuple[str, str]] = []


def open_db(path: Path) -> sqlite3.Connection:
    return open_log_db(
        path, [_SCHEMA, _RUNS_SCHEMA],
        {"jumps": _LATE_COLUMNS, "runs": _RUNS_LATE_COLUMNS},
    )


def log_run(conn: sqlite3.Connection, **fields) -> int:
    """Insert a per-attempt run-summary row (#6: game-over + final score).
    Returns the rowid."""
    return insert_row(conn, "runs", fields)


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
                              kind: str = "pit",
                              source: str | None = "bot",
                              ) -> list[tuple[int, int, int]]:
    """Return [(distance_bin, n_total, n_survived), ...] for jumps of
    the given kind, grouped into 10-px distance bins. Quick way to see
    'jumps fired at pit_dist=50 survived 8/10, pit_dist=70 survived
    2/10 — tighten the trigger window.'

    source defaults to 'bot' so the bot's [30,50] policy isn't blended with
    human --watch clicks (different rhythm/aim) — mixing them describes
    neither policy and contaminates the trigger-window calibration. Pass
    source=None to include all rows, or 'human' for the watch data. Only
    settled rows (outcome in survived/died) count; legacy 'unknown' and
    still-pending rows are excluded so the ratio is meaningful."""
    where = ["next_kind = ?", "next_distance_px IS NOT NULL",
             "outcome IN ('survived', 'died')"]
    params: list = [kind]
    if source is not None:
        where.append("source = ?")
        params.append(source)
    rows = conn.execute(
        f'''
        SELECT (next_distance_px / 10) * 10 AS bin,
               COUNT(*),
               SUM(CASE WHEN outcome = 'survived' THEN 1 ELSE 0 END)
        FROM jumps
        WHERE {" AND ".join(where)}
        GROUP BY bin
        ORDER BY bin
        ''',
        params,
    ).fetchall()
    return [(int(b), int(n), int(s or 0)) for b, n, s in rows]
