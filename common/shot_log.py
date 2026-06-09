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
    source TEXT,
    rim_match_scale REAL,
    time_since_last_shot_ms INTEGER,
    ball_flight_ms INTEGER,
    bob_ymin INTEGER,
    bob_ymax INTEGER,
    bob_period_ms INTEGER,
    platform_vy REAL,
    made_source TEXT
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
    # Diagnostic columns added in the metrics-backlog batch (closes #11,
    # #12, #14, #16). All optional, NULL on rows logged before they
    # existed. See the corresponding issues for rationale.
    ("rim_match_scale", "REAL"),         # multi-scale rim template match scale
    ("time_since_last_shot_ms", "INTEGER"),  # gap between this fired_at and the previous
    ("ball_flight_ms", "INTEGER"),       # first-detected to last-detected across flight frames
    ("bob_ymin", "INTEGER"),             # observed platform_y range at fire time
    ("bob_ymax", "INTEGER"),
    ("bob_period_ms", "INTEGER"),        # mean ms between consecutive bob peaks (#13)
    # Rightmost ball x during flight. Compared against ball_landing_x in
    # clean_make: peak_x >> landing_x means the ball travelled past the
    # hoop and bounced back (off rim/backboard), not a clean shot. Added
    # 2026-05-23 after discovering ~25% of historical "clean makes" were
    # actually backboard-bounce-ins that the landing-x-only filter missed.
    ("ball_peak_x", "INTEGER"),
    # Platform vertical velocity (px/s, positive = downward in screen
    # coords) at fire time, least-squares slope over the platform samples
    # in the ~300ms before the click. Added 2026-06-09 (#37): misses are
    # bimodal (way short / way long) at identical (hoop, offset), and the
    # suspected hidden variable is the platform velocity the ball inherits
    # at launch — this column is the instrument to confirm or refute that.
    ("platform_vy", "REAL"),
    # How the make verdict on this row was reached, when it wasn't the
    # standard OCR-increment path. Currently only "respawn": the hoop
    # teleported after a shot the OCR couldn't confirm — the hoop only
    # respawns on makes, so the shot is retroactively marked made
    # (added 2026-06-09, #37: OCR dropouts were losing ~10% of real
    # makes). NULL on rows where OCR confirmed the make directly.
    ("made_source", "TEXT"),
]


# Tolerance (px) on |ball_landing_x - hoop_x| for declaring a make "clean".
# Empirically: confirmed swishes land within ~30-40px of hoop_x; rattle-ins
# can end up 150-200px off because the last-detected ball position catches
# a bounce off the backboard.
CLEAN_MAKE_TOLERANCE = 80

# Maximum backward-drift (ball_peak_x - ball_landing_x) for a make to count
# as clean. A clean swish or near-swish: ball goes up, comes down, lands
# near where it peaked → drift ≈ 0. A backboard ricochet: ball flies past
# the hoop, hits backboard post, deflects back and falls through → peak_x
# can be hundreds of px past landing_x. Threshold chosen 2026-05-23 from a
# 30-row sample of "clean" makes where the bimodal distribution showed
# clean shots clustered at drift 0-30 and bounces at 100-290. 50 sits
# comfortably between the modes.
MAX_BACK_DRIFT_PX = 50


# Minimum (ball_landing_x - platform_x) for a shot's trajectory to be
# considered "clean enough" to learn from. Empirically the miss landing
# distribution is sharply bimodal: rim/backboard bounces deflect the ball
# back and it lands 0-300 px right of the launcher, while shots that
# actually flew toward the hoop land 400+ px out. Trajectory predictors
# trained on the bounce cluster learn nonsense; gating at 300 cleanly
# separates the two modes (verified against the historical shots.db,
# 2026-05-09). See GitHub issue #18.
MIN_TRAJECTORY_DISTANCE_PX = 300


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
    # - made = 1 + ball_landing_x within tolerance of hoop_x AND
    #   (peak_x NULL OR peak_x - landing_x within back-drift limit) → clean
    # - made = 1 + ball_landing_x is NULL → trust the make (historical
    #   data without trajectory analysis available; not punishing it)
    # - made = 1 + ball_landing_x outside tolerance → not clean
    # - made = 1 + peak_x - landing_x > back-drift limit → not clean (bounce)
    # - made = 0 → clean_make = 0
    #
    # IS NULL filter preserves the "explicit caller values stick" contract.
    # Historical reclassification under newer rules is the backfill script's
    # job (scripts/backfill_peak_x.py), not _migrate's.
    conn.execute(
        '''
        UPDATE shots
           SET clean_make = CASE
             WHEN made IS NOT 1 THEN 0
             WHEN ball_landing_x IS NULL OR hoop_x IS NULL THEN 1
             WHEN ABS(ball_landing_x - hoop_x) > ? THEN 0
             WHEN ball_peak_x IS NOT NULL
                  AND (ball_peak_x - ball_landing_x) > ? THEN 0
             ELSE 1
           END
         WHERE clean_make IS NULL
        ''',
        (CLEAN_MAKE_TOLERANCE, MAX_BACK_DRIFT_PX),
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


def log_shot(conn: sqlite3.Connection, **fields) -> int:
    """Insert a shot row and return its rowid (so outcome corrections can
    target it later). Caller passes whichever columns they have; the rest
    default to NULL. `offset` is a SQLite-quoted column name."""
    cols = ", ".join(f'"{k}"' if k == "offset" else k for k in fields)
    placeholders = ", ".join("?" * len(fields))
    cur = conn.execute(
        f"INSERT INTO shots ({cols}) VALUES ({placeholders})",
        tuple(fields.values()),
    )
    conn.commit()
    return cur.lastrowid


def set_make(
    conn: sqlite3.Connection,
    shot_id: int,
    made: int,
    clean_make: int | None,
    made_source: str | None = None,
) -> None:
    """Retroactively correct a shot's make verdict (e.g. when the hoop
    respawn proves a shot made after OCR failed to confirm it)."""
    conn.execute(
        "UPDATE shots SET made = ?, clean_make = ?, made_source = ? WHERE id = ?",
        (made, clean_make, made_source, shot_id),
    )
    conn.commit()


def fetch_makes(
    conn: sqlite3.Connection,
    required_direction: str,
) -> list[tuple[float, float, float]]:
    """Return [(hoop_y, hoop_x, platform_y), ...] for clean makes in the
    requested required_direction.

    Caller hands these to a `fit_*` factory in common.predictor.

    Filters on `clean_make = 1` rather than `made = 1`: rattle-ins (ball
    bounced off the backboard, then trickled in) inflate the make count
    but their platform_y at fire time is a poor target to learn from,
    since the ball's actual flight was off-aim. See `_migrate` for the
    rule that classifies a make as clean.

    Clamped rows are kept (#20). Earlier this query excluded them on the
    rationale that "fired at bob extreme regardless of target_y", but
    the logged `platform_y` is the actual fire value — same factual
    signal as any other row. Excluding clamped makes was preventing
    the predictor from learning when the perturbation sweep discovered
    a working offset at the bob limit.
    """
    return [
        (float(r[0]), float(r[1]), float(r[2]))
        for r in conn.execute(
            'SELECT hoop_y, hoop_x, platform_y FROM shots '
            'WHERE clean_make = 1 AND required_direction = ? '
            'AND hoop_y IS NOT NULL AND hoop_x IS NOT NULL '
            'AND platform_y IS NOT NULL',
            (required_direction,),
        )
    ]


def fetch_shots(
    conn: sqlite3.Connection,
    required_direction: str,
) -> list[tuple[float, float, float, float]]:
    """Return [(hoop_y, hoop_x, platform_y, ball_landing_x), ...] for every
    shot with a tracked trajectory in the requested required_direction.
    Excludes rows where any input or the trajectory landing is missing.

    Unlike fetch_makes, this includes misses — the trajectory predictor
    learns from where any shot lands, not just the makes. Used by
    `fit_trajectory_knn` and friends in common.predictor.

    No bounce filtering is applied — analysis tooling sometimes needs the
    raw trajectories. Predictors should call `fetch_clean_trajectories`
    instead. Clamped rows are included (see fetch_makes for rationale).
    """
    return [
        (float(r[0]), float(r[1]), float(r[2]), float(r[3]))
        for r in conn.execute(
            'SELECT hoop_y, hoop_x, platform_y, ball_landing_x FROM shots '
            'WHERE required_direction = ? '
            'AND hoop_y IS NOT NULL AND hoop_x IS NOT NULL '
            'AND platform_y IS NOT NULL AND ball_landing_x IS NOT NULL',
            (required_direction,),
        )
    ]


# Backward deflections that happened AT the hoop censor the flight's true
# reach (the structure stopped the ball) — those rows can't train a reach
# model. Deflections far from the hoop are floor bounces: the rightmost x
# the ball reached is still an honest reach measurement.
HOOP_DEFLECTION_ZONE_PX = 80


def _trajectory_arrival(
    hoop_x: float,
    landing_x: float,
    peak_x: float | None,
    platform_x: float,
    min_distance_px: int,
) -> float | None:
    """Reach of a tracked flight, or None when the row can't train a
    trajectory model.

    - No backward drift (peak ≈ landing): clean flight, reach = landing.
    - Backward drift with peak near hoop_x: the ball deflected off the
      hoop structure — its unobstructed reach is unknowable, drop it.
    - Backward drift far from the hoop: floor bounce — the ball landed
      short and rolled/bounced back; peak_x is the honest reach. (The
      legacy distance-only filter dropped these, which threw away the
      short-mislaunch population carrying the velocity signal, #37.)
    - peak_x missing (rows before 2026-05-23): legacy heuristic — keep
      only flights landing at least min_distance_px from the launcher.
    """
    if peak_x is None:
        return landing_x if (landing_x - platform_x) >= min_distance_px else None
    if (peak_x - landing_x) <= MAX_BACK_DRIFT_PX:
        return landing_x
    if abs(peak_x - hoop_x) <= HOOP_DEFLECTION_ZONE_PX:
        return None
    return peak_x


_TRAJECTORY_BASE_SQL = (
    'SELECT hoop_y, hoop_x, platform_y, {extra} ball_landing_x, ball_peak_x, platform_x '
    'FROM shots WHERE required_direction = ? '
    'AND hoop_y IS NOT NULL AND hoop_x IS NOT NULL '
    'AND platform_y IS NOT NULL AND ball_landing_x IS NOT NULL '
    'AND platform_x IS NOT NULL {extra_filter}'
)


def fetch_clean_trajectories(
    conn: sqlite3.Connection,
    required_direction: str,
    min_distance_px: int = MIN_TRAJECTORY_DISTANCE_PX,
) -> list[tuple[float, float, float, float]]:
    """Rows of (hoop_y, hoop_x, platform_y, arrival_x) for trajectory
    regression, where arrival_x is the flight's reach per
    _trajectory_arrival — bounce-aware, hoop-clank rows excluded.
    Clamped rows are included (see fetch_makes for rationale).
    """
    out: list[tuple[float, float, float, float]] = []
    for hy, hx, py, landing, peak, px in conn.execute(
        _TRAJECTORY_BASE_SQL.format(extra='', extra_filter=''),
        (required_direction,),
    ):
        arrival = _trajectory_arrival(hx, landing, peak, px, min_distance_px)
        if arrival is not None:
            out.append((float(hy), float(hx), float(py), float(arrival)))
    return out


def fetch_clean_trajectories_vy(
    conn: sqlite3.Connection,
    required_direction: str,
    min_distance_px: int = MIN_TRAJECTORY_DISTANCE_PX,
) -> list[tuple[float, float, float, float, float]]:
    """Like fetch_clean_trajectories but returns
    (hoop_y, hoop_x, platform_y, platform_vy, arrival_x) for rows where
    the platform velocity at fire was recorded. Training data for the
    velocity-aware trajectory predictor (#37)."""
    out: list[tuple[float, float, float, float, float]] = []
    for hy, hx, py, vy, landing, peak, px in conn.execute(
        _TRAJECTORY_BASE_SQL.format(
            extra='platform_vy,', extra_filter='AND platform_vy IS NOT NULL',
        ),
        (required_direction,),
    ):
        arrival = _trajectory_arrival(hx, landing, peak, px, min_distance_px)
        if arrival is not None:
            out.append((float(hy), float(hx), float(py), float(vy), float(arrival)))
    return out
