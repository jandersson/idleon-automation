"""SQLite throw log for the darts bot.

Mirrors the shape of the hoops shot log (minigames/hoops/shot_log.py)
but with a darts-specific schema:
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

from common.db_log import open_log_db, insert_row


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
# Keep this list append-only — same pattern as the hoops shot log.
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
    # Dart vertical displacement between the previous frame and the firing
    # match — first #26 discriminator candidate, DEAD ON ARRIVAL: the dart
    # rotates between polls at the ~250ms cadence, so the release-angle
    # template never re-matched the prev frame (NULL on every live fire,
    # 2026-06-10). Column kept dormant per append-only convention.
    ("dart_dy_at_fire", "INTEGER"),
    # Arm-centroid dy vs the previous poll at fire time — the #26
    # discriminator that actually separates. Validated post-hoc against
    # launch-angle ground truth (23 fires, 2026-06-10): dy < 0 → 9/10
    # hits, dy > 0 → 10/11 misses. Logged at fire so the eventual gate
    # uses exactly the quantity the validation measured.
    ("arm_centroid_dy_at_fire", "INTEGER"),
    # How the dy-band aim decided this fire (#41): 'band' (inside the EV
    # band), 'fallback' (skip budget exhausted or dy unavailable), or
    # 'explore' (ε-exploration throw). Lets the gate's EV be evaluated
    # per intent and keeps model training aware of the sampling policy.
    ("aim_mode", "TEXT"),
    # Numeric wind (#41 step 2), parsed from the panel at fire time:
    # speed in mph; angle in degrees (screen convention, 0 = blowing
    # right, +90 = down); wind_x/wind_y = speed * (cos, sin) — the
    # model features. Calm = (0, NULL, 0, 0); parse failure = all NULL.
    # Encoded as components rather than (speed, direction) because RBF
    # kernels can't see around the angle wraparound.
    ("wind_speed", "REAL"),
    ("wind_angle_deg", "REAL"),
    ("wind_x", "REAL"),
    ("wind_y", "REAL"),
    # Arm-centroid vertical velocity at fire in px/SECOND — the
    # cadence-invariant replacement for arm_centroid_dy_at_fire
    # (px/poll at the ~250ms compute-bound cadence; dormant from
    # 2026-06-10, kept per append-only convention). The unit change
    # unblocks poll-loop speedups: per-poll units rescale with every
    # tick-cost optimization. Convert legacy rows via each fire's
    # actual poll gap (see stripe_model.fetch_stripe_rows).
    ("arm_centroid_vy_at_fire", "INTEGER"),
]


_POLLS_LATE_COLUMNS: list[tuple[str, str]] = [
    # Per-poll arm-motion centroid added 2026-05-24 for #25 discriminator
    # work. See _POLLS_SCHEMA comment for semantics.
    ("arm_centroid_y", "INTEGER"),
    ("arm_pixel_count", "INTEGER"),
    # Dart dy vs the previous frame, computed only on polls whose template
    # match cleared RELEASE_THRESHOLD (#26). NULL elsewhere.
    ("dart_dy", "INTEGER"),
]


def open_db(path: Path) -> sqlite3.Connection:
    return open_log_db(
        path, [_SCHEMA, _POLLS_SCHEMA],
        {"throws": _LATE_COLUMNS, "polls": _POLLS_LATE_COLUMNS},
    )


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
    dart_dy: int | None = None,
) -> None:
    """Append one row to the polls table. Called once per main-loop
    iteration. No commit per call — caller commits periodically (e.g.
    every fire) to keep write overhead bounded."""
    conn.execute(
        "INSERT INTO polls (session_started, t_ms, conf, match_x, match_y, threw, "
        "arm_centroid_y, arm_pixel_count, dart_dy) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            session_started, int(t_ms), float(conf), int(match_x), int(match_y), int(threw),
            int(arm_centroid_y) if arm_centroid_y is not None else None,
            int(arm_pixel_count) if arm_pixel_count is not None else None,
            int(dart_dy) if dart_dy is not None else None,
        ),
    )


def log_throw(conn: sqlite3.Connection, **fields) -> int:
    """Insert a throw row. Caller passes whichever columns they have;
    the rest default to NULL. Returns the rowid."""
    return insert_row(conn, "throws", fields)


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
