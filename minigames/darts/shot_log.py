"""SQLite throw log for the darts bot.

Mirrors the shape of common.shot_log but with a darts-specific schema:
release pose, conf, launch angle / apex / landing from
common.dart_trajectory, score increment, wind sample id, and the
streak counter at log time.

Per-throw rows let us answer questions like "for wind state X, what
launch angle has the highest hit rate?" — the basis for any future
wind-conditioned release-angle predictor.

Usage:
    from minigames.darts.shot_log import open_db, log_throw
    conn = open_db(Path("minigames/darts/assets/darts.db"))
    log_throw(conn, session_started="...", throw_idx=1, hit=1, ...)
    conn.close()
"""
import sqlite3
from pathlib import Path


_SCHEMA = """
CREATE TABLE IF NOT EXISTS throws (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_started TEXT,
    throw_idx INTEGER,
    fired_at TEXT,
    release_pose_x INTEGER,
    release_pose_y INTEGER,
    release_conf REAL,
    wind_sample TEXT,
    launch_angle_deg REAL,
    apex_y INTEGER,
    landing_x INTEGER,
    frames_seen INTEGER,
    score_before_int INTEGER,
    score_after_int INTEGER,
    score_increment INTEGER,
    hit INTEGER,
    bullseye INTEGER,
    streak INTEGER,
    throw_dir TEXT,
    window_w INTEGER,
    window_h INTEGER,
    code_commit TEXT,
    source TEXT
)
"""


# New columns added after the original schema. open_db() runs ALTER TABLE
# for each on existing DBs (sqlite ignores duplicate-column errors).
# Keep this list append-only — same pattern as common.shot_log.
_LATE_COLUMNS: list[tuple[str, str]] = [
    # (no late columns yet — placeholder for future additions)
]


def _migrate(conn: sqlite3.Connection) -> None:
    for name, decl in _LATE_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE throws ADD COLUMN {name} {decl}")
        except sqlite3.OperationalError:
            pass


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute(_SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


def log_throw(conn: sqlite3.Connection, **fields) -> None:
    """Insert a throw row. Caller passes whichever columns they have;
    the rest default to NULL."""
    cols = ", ".join(fields)
    placeholders = ", ".join("?" * len(fields))
    conn.execute(
        f"INSERT INTO throws ({cols}) VALUES ({placeholders})",
        tuple(fields.values()),
    )
    conn.commit()


def fetch_hits(
    conn: sqlite3.Connection,
    only_with_trajectory: bool = True,
) -> list[tuple[float, int, int]]:
    """Return [(launch_angle_deg, landing_x, score_increment), ...] for
    rows that we'd want to train a release-angle predictor on.

    `only_with_trajectory=True` (default) requires launch_angle_deg
    and landing_x to be non-NULL — i.e. the trajectory module
    actually saw the dart. Misses ARE included so the predictor can
    learn from off-aim throws too; it's hits-AND-misses-with-data,
    not just-hits.
    """
    sql = (
        "SELECT launch_angle_deg, landing_x, score_increment "
        "FROM throws "
        "WHERE launch_angle_deg IS NOT NULL AND landing_x IS NOT NULL"
    ) if only_with_trajectory else (
        "SELECT launch_angle_deg, landing_x, score_increment FROM throws"
    )
    return [
        (float(r[0]) if r[0] is not None else None,
         int(r[1]) if r[1] is not None else None,
         int(r[2]) if r[2] is not None else None)
        for r in conn.execute(sql)
    ]
