"""SQLite shot log for tuning timing-based bots.

The hoops bot writes one row per shot with hoop position, offset, platform
state at fire time, score diff, and a path to the per-shot monitor folder.
That makes it easy to query "what offset worked at hoop_y=X" or "show all
makes where direction=up" instead of grepping log files.

Usage:
    from common.shot_log import open_db, log_shot
    conn = open_db(Path("minigames/hoops/assets/shots.db"))
    log_shot(conn, session_started="...", shot_idx=1, hoop_x=710, ...)
    conn.close()

Querying: `sqlite3 minigames/hoops/assets/shots.db` and run SQL.
"""
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS shots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_started TEXT,
    shot_idx INTEGER,
    fired_at TEXT,
    hoop_x INTEGER,
    hoop_y INTEGER,
    hoop_conf REAL,
    platform_x INTEGER,
    platform_y INTEGER,
    "offset" INTEGER,
    target_y INTEGER,
    eff_target_y INTEGER,
    clamped INTEGER,
    direction TEXT,
    required_direction TEXT,
    score_diff REAL,
    made INTEGER,
    shot_dir TEXT
)
"""


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def log_shot(conn: sqlite3.Connection, **fields) -> None:
    """Insert a shot row. Caller passes whichever columns they have; the rest
    default to NULL. `offset` is a SQLite-quoted column name."""
    cols = ", ".join(f'"{k}"' if k == "offset" else k for k in fields)
    placeholders = ", ".join("?" * len(fields))
    conn.execute(
        f"INSERT INTO shots ({cols}) VALUES ({placeholders})",
        tuple(fields.values()),
    )
    conn.commit()


def fit_target_predictor(
    conn: sqlite3.Connection,
    required_direction: str,
    min_samples: int = 3,
) -> tuple[float, float, int] | None:
    """Linear regression `optimal_platform_y = m*hoop_y + b` from confirmed makes.

    Fits on the *observed* platform_y at fire time (what actually launched
    successful shots), not the bot's nominal target_y. The two diverge
    because the bot's "crossed" detection fires the sample after the actual
    cross, so platform_y is consistently below target_y on the upstroke by
    a sampling-rate-dependent amount. Caller adds an UPSTROKE_COMPENSATION
    constant to convert from optimal_platform_y back to target_y.

    Excludes clamped shots — those fired at the bob extreme regardless of
    nominal target_y, so they say nothing about timing.

    Returns (m, b, n_samples) or None when there isn't enough data.
    """
    rows = list(
        conn.execute(
            'SELECT hoop_y, platform_y FROM shots '
            'WHERE made = 1 AND clamped = 0 AND required_direction = ? '
            'AND hoop_y IS NOT NULL AND platform_y IS NOT NULL',
            (required_direction,),
        )
    )
    n = len(rows)
    if n < min_samples:
        return None
    xs = [float(r[0]) for r in rows]
    ys = [float(r[1]) for r in rows]
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return None
    m = num / den
    b = mean_y - m * mean_x
    return m, b, n
