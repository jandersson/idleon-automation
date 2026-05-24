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


# Per-poll diagnostic log. The bot writes a row every iteration of its
# main loop, regardless of whether the template matched. Lets us reconstruct
# the full conf/y trajectory around each fire post-hoc, with the throws
# table providing hit/miss ground truth. Built 2026-05-24 to fix the
# "fire happens but we can't tell which arm-swing pass we caught" problem.
#
# threw=1 indicates this poll triggered a click (the matching row in
# throws will have the same session_started and a fired_at within a few
# ms of t_ms's wall-clock equivalent). Most rows have threw=0.
_POLLS_SCHEMA = """
CREATE TABLE IF NOT EXISTS polls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_started TEXT,
    t_ms INTEGER,       -- milliseconds since session start
    conf REAL,          -- template-match conf (unthresholded)
    match_x INTEGER,    -- center x of the best template match
    match_y INTEGER,    -- center y of the best template match
    threw INTEGER,      -- 1 if this poll fired a click, 0 otherwise
    -- Arm centroid added 2026-05-24 (#25 work): motion-AND-white mask
    -- vs the previous poll's frame, isolates the swinging arm pixels.
    -- centroid_y high (small value) = arm cocked back at apex; low = arm
    -- extended forward at release. Continuous signal independent of
    -- template-match timing — should discriminate the two firing
    -- moments where conf/match_y alone failed.
    arm_centroid_y INTEGER,
    arm_pixel_count INTEGER
)
"""


# New columns added after the original schema. open_db() runs ALTER TABLE
# for each on existing DBs (sqlite ignores duplicate-column errors).
# Keep this list append-only — same pattern as common.shot_log.
_LATE_COLUMNS: list[tuple[str, str]] = [
    # Temporal diagnostics added 2026-05-24 to distinguish forward-release
    # (hit) from top-of-swing apex (miss). Both moments produce a template
    # match, so the bot can't tell them apart from a single frame — but
    # the apex match holds for many consecutive poll frames while forward
    # release passes through quickly. Capture both signals per throw so
    # we can validate the hypothesis from labeled hit/miss data.
    ("match_streak_len_before_fire", "INTEGER"),  # consecutive matches incl. the firing frame
    ("prev_match_y", "INTEGER"),  # dart y on the prior matched frame this streak; NULL if streak_len=1
    # Stripe color hit, derived from score_increment via STRIPE_COLOR_BY_INCREMENT
    # in main.py. NULL when score OCR failed. Lets us slice analytics by
    # color without re-decoding score_increment each query, and surfaces
    # color shifts in trajectory data (e.g. landing_x vs stripe).
    ("stripe_color", "TEXT"),
]


_POLLS_LATE_COLUMNS: list[tuple[str, str]] = [
    # Per-poll arm-motion centroid added 2026-05-24 for #25 discriminator
    # work. See _POLLS_SCHEMA comment for semantics.
    ("arm_centroid_y", "INTEGER"),
    ("arm_pixel_count", "INTEGER"),
]


def _migrate(conn: sqlite3.Connection) -> None:
    for name, decl in _LATE_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE throws ADD COLUMN {name} {decl}")
        except sqlite3.OperationalError:
            pass
    for name, decl in _POLLS_LATE_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE polls ADD COLUMN {name} {decl}")
        except sqlite3.OperationalError:
            pass


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute(_SCHEMA)
    conn.execute(_POLLS_SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


def log_poll(
    conn: sqlite3.Connection,
    session_started: str,
    t_ms: int,
    conf: float,
    match_x: int,
    match_y: int,
    threw: int,
    arm_centroid_y: int | None = None,
    arm_pixel_count: int | None = None,
) -> None:
    """Append one row to the polls table. Called once per main-loop
    iteration. No commit per call — caller commits periodically (e.g.
    every fire) to keep write overhead bounded."""
    conn.execute(
        "INSERT INTO polls (session_started, t_ms, conf, match_x, match_y, threw, "
        "arm_centroid_y, arm_pixel_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            session_started, int(t_ms), float(conf), int(match_x), int(match_y), int(threw),
            int(arm_centroid_y) if arm_centroid_y is not None else None,
            int(arm_pixel_count) if arm_pixel_count is not None else None,
        ),
    )


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
