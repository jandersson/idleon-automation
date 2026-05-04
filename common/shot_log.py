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
import subprocess
import sqlite3
from pathlib import Path


def current_code_commit(repo_root: Path) -> str | None:
    """Return the short git commit hash of HEAD, with "-dirty" suffix if
    the working tree has uncommitted changes. None if not in a git repo
    or git isn't available."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return None

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
    shot_dir TEXT,
    perturbation INTEGER,
    lives_diff REAL,
    ball_apex_y INTEGER,
    ball_x_at_rim_height INTEGER,
    ball_landing_x INTEGER,
    window_w INTEGER,
    window_h INTEGER,
    score_before_int INTEGER,
    score_after_int INTEGER,
    score_increment INTEGER,
    predicted_offset INTEGER,
    code_commit TEXT,
    predictor_kind TEXT,
    source TEXT
)
"""

# New columns added after the original schema. open_db() runs ALTER TABLE
# for each on existing DBs (SQLite ignores duplicate-column errors). Keep
# this list append-only.
_LATE_COLUMNS = [
    # click_x / click_y were here, dropped from the schema after click
    # position was proven not to matter (May 3). Old DBs keep the
    # columns harmlessly; new DBs don't get them.
    ("perturbation", "INTEGER"),
    ("lives_diff", "REAL"),
    ("ball_apex_y", "INTEGER"),
    ("ball_x_at_rim_height", "INTEGER"),
    ("ball_landing_x", "INTEGER"),
    ("window_w", "INTEGER"),
    ("window_h", "INTEGER"),
    ("score_before_int", "INTEGER"),
    ("score_after_int", "INTEGER"),
    ("score_increment", "INTEGER"),
    ("predicted_offset", "INTEGER"),
    ("code_commit", "TEXT"),
    ("predictor_kind", "TEXT"),
    ("source", "TEXT"),  # "bot" (default) or "human" (hoops-observe mode)
    # Stricter "predictor-quality" make filter: a shot was a make AND the
    # ball trajectory looks consistent with the aim (vs. lucky rattle-in).
    # Used by fetch_makes to avoid contaminating predictor training data.
    ("clean_make", "INTEGER"),
]


# Tolerance (px) on |ball_landing_x - hoop_x| for declaring a make "clean".
# Empirically: confirmed swishes land within ~30-40px of hoop_x; rattle-ins
# can end up 150-200px off because the last-detected ball position catches
# a bounce off the backboard.
CLEAN_MAKE_TOLERANCE = 80


def _migrate(conn: sqlite3.Connection) -> None:
    """Add any new columns to an existing shots table that didn't have them.
    Then backfill `clean_make` for any made-row that's missing it, using
    the same rule we apply at log time."""
    for name, decl in _LATE_COLUMNS:
        try:
            conn.execute(f'ALTER TABLE shots ADD COLUMN {name} {decl}')
        except sqlite3.OperationalError:
            # Already exists — duplicate column error.
            pass

    # Backfill clean_make on existing rows. Rule:
    # - made = 1 + ball_landing_x within tolerance of hoop_x → clean
    # - made = 1 + ball_landing_x is NULL → trust the make (historical
    #   data without trajectory analysis available; not punishing it)
    # - made = 1 + ball_landing_x outside tolerance → not clean
    # - made = 0 → clean_make = 0
    conn.execute(
        '''
        UPDATE shots
           SET clean_make = CASE
             WHEN made IS NOT 1 THEN 0
             WHEN ball_landing_x IS NULL OR hoop_x IS NULL THEN 1
             WHEN ABS(ball_landing_x - hoop_x) <= ? THEN 1
             ELSE 0
           END
         WHERE clean_make IS NULL
        ''',
        (CLEAN_MAKE_TOLERANCE,),
    )


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False so observe-mode's threading.Timer callbacks
    # (post-shot OCR + log_shot, fired from a worker thread) can use the
    # same connection. SQLite serialises writes internally so concurrent
    # log_shot calls from multiple threads are safe.
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute(SCHEMA)
    _migrate(conn)
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


def fetch_makes(
    conn: sqlite3.Connection,
    required_direction: str,
) -> list[tuple[float, float, float]]:
    """Return [(hoop_y, hoop_x, platform_y), ...] for clean makes in the
    requested required_direction. Excludes clamped shots — those fired at
    the bob extreme regardless of nominal target_y, so they don't reflect
    the predictor's signal.

    Caller hands these to a `fit_*` factory in common.predictor.

    Filters on `clean_make = 1` rather than `made = 1`: rattle-ins (ball
    bounced off the backboard, then trickled in) inflate the make count
    but their platform_y at fire time is a poor target to learn from,
    since the ball's actual flight was off-aim. See `_migrate` for the
    rule that classifies a make as clean.
    """
    return [
        (float(r[0]), float(r[1]), float(r[2]))
        for r in conn.execute(
            'SELECT hoop_y, hoop_x, platform_y FROM shots '
            'WHERE clean_make = 1 AND clamped = 0 AND required_direction = ? '
            'AND hoop_y IS NOT NULL AND hoop_x IS NOT NULL '
            'AND platform_y IS NOT NULL',
            (required_direction,),
        )
    ]
