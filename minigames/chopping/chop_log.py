"""SQLite chop log for the chopping bot.

One row per click decision: detector state at fire time (pointer_x in the
bar, zone under the leaf's left edge, distance to the nearest red column,
the safety margin we applied) + an outcome measured on the next loop
iteration ('survived' if the bot's still chopping, 'game_over' if the
run ended).

The "button click is suspect" TODO sits unanswered because session logs
only record what the detector *thought* at fire time — not whether the
chop landed safely. With this, a single session yields the data to ask
e.g. "of clicks where nearest_red_distance was 9-12 px, what fraction
ended the run?" — directly testing whether RED_SAFETY_MARGIN_PX is too
tight, or whether leaf-edge detection is off.

Usage:
    from minigames.chopping.chop_log import open_db, log_chop, set_outcome
    conn = open_db(Path("minigames/chopping/assets/chopping.db"))
    row_id = log_chop(conn, session_started="...", chop_idx=1, ...)
    set_outcome(conn, row_id, "survived", measured_ms=460)
"""
import sqlite3
from pathlib import Path

from common.db_log import open_log_db, insert_row, update_row


_SCHEMA = """
CREATE TABLE IF NOT EXISTS chops (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_started TEXT,
    chop_idx INTEGER,
    clicked_at TEXT,
    pointer_x INTEGER,
    zone TEXT,
    nearest_red_distance INTEGER,
    red_safety_margin INTEGER,
    bar_left INTEGER,
    bar_top INTEGER,
    bar_width INTEGER,
    bar_height INTEGER,
    button_click_x INTEGER,
    button_click_y INTEGER,
    window_w INTEGER,
    window_h INTEGER,
    outcome TEXT,
    outcome_measured_ms INTEGER,
    code_commit TEXT,
    source TEXT
)
"""


# Per-poll diagnostic log. The bot writes a row every loop iteration
# (sub-sampled to keep write rate sane), regardless of whether a click
# fired. Lets us reconstruct what the bar looked like *between* chops —
# pointer x, zone under the leaf, distance to red, total colored bar
# pixels. The chops table provides ground truth for the click
# decisions, polls provides the surrounding context.
#
# fired=1 indicates this poll triggered a click (the matching row in
# chops has the same session_started and a clicked_at within a few ms
# of t_ms's wall-clock equivalent). Most rows have fired=0.
_POLLS_SCHEMA = """
CREATE TABLE IF NOT EXISTS polls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_started TEXT,
    t_ms INTEGER,
    pointer_x INTEGER,
    zone TEXT,
    nearest_red_distance INTEGER,
    bar_pixel_count INTEGER,
    fired INTEGER,
    zone_layout TEXT
)
"""


_LATE_COLUMNS: list[tuple[str, str]] = [
    # (no late columns yet — placeholder for future additions)
]


_POLLS_LATE_COLUMNS: list[tuple[str, str]] = [
    # RLE of the bar's per-column zone colors ("n4 g58 r12 ..."), see
    # detector.zone_layout. Added 2026-06-10 before the polls table ever
    # saw a live session, so the ALTER is a no-op everywhere in practice.
    ("zone_layout", "TEXT"),
]


def open_db(path: Path) -> sqlite3.Connection:
    return open_log_db(
        path, [_SCHEMA, _POLLS_SCHEMA],
        {"chops": _LATE_COLUMNS, "polls": _POLLS_LATE_COLUMNS},
    )


def log_poll(
    conn: sqlite3.Connection,
    session_started: str,
    t_ms: int,
    pointer_x: int | None,
    zone: str,
    nearest_red_distance: int | None,
    bar_pixel_count: int,
    fired: int,
    zone_layout: str | None = None,
) -> None:
    """Append one row to the polls table. No per-row commit — caller
    commits periodically (e.g. once per fire) to bound write overhead."""
    conn.execute(
        "INSERT INTO polls (session_started, t_ms, pointer_x, zone, "
        "nearest_red_distance, bar_pixel_count, fired, zone_layout) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            session_started, int(t_ms),
            int(pointer_x) if pointer_x is not None else None,
            zone,
            int(nearest_red_distance) if nearest_red_distance is not None else None,
            int(bar_pixel_count), int(fired),
            zone_layout,
        ),
    )


def log_chop(conn: sqlite3.Connection, **fields) -> int:
    """Insert a chop row, return its rowid so callers can set_outcome
    on it later."""
    return insert_row(conn, "chops", fields)


def set_outcome(conn: sqlite3.Connection, row_id: int, outcome: str,
                measured_ms: int) -> None:
    """Fill in outcome + outcome_measured_ms on an existing chop row."""
    update_row(conn, "chops", row_id,
               {"outcome": outcome, "outcome_measured_ms": measured_ms})


def outcome_rate_by_red_distance(
    conn: sqlite3.Connection,
    bin_px: int = 4,
) -> list[tuple[int, int, int, int]]:
    """Return [(distance_bin, n_total, n_survived, n_game_over), ...]
    grouped into `bin_px`-px bins of nearest_red_distance.

    Quick way to see "at red_dist=8-11 px the run died 4/12 times,
    at 12-15 px 0/30 — tighten RED_SAFETY_MARGIN_PX to 12." Only
    counts rows with a non-NULL distance (red was present in the
    bar) and a settled outcome."""
    rows = conn.execute(
        f"""
        SELECT (nearest_red_distance / {bin_px}) * {bin_px} AS bin,
               COUNT(*),
               SUM(CASE WHEN outcome = 'survived'  THEN 1 ELSE 0 END),
               SUM(CASE WHEN outcome = 'game_over' THEN 1 ELSE 0 END)
        FROM chops
        WHERE nearest_red_distance IS NOT NULL
          AND outcome IN ('survived', 'game_over')
        GROUP BY bin
        ORDER BY bin
        """,
    ).fetchall()
    return [(int(b), int(n), int(s or 0), int(g or 0)) for b, n, s, g in rows]
