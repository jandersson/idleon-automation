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
    code_commit TEXT
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
]


def _migrate(conn: sqlite3.Connection) -> None:
    """Add any new columns to an existing shots table that didn't have them."""
    for name, decl in _LATE_COLUMNS:
        try:
            conn.execute(f'ALTER TABLE shots ADD COLUMN {name} {decl}')
        except sqlite3.OperationalError:
            # Already exists — duplicate column error.
            pass


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
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


class KnnPredictor:
    """KNN-over-past-makes predictor. Each make is a point in
    (hoop_y, hoop_x) space, and its `platform_y` is the optimal fire
    position for that hoop. To predict for a new hoop, take the K
    closest past makes and inverse-distance-weight their platform_y.

    KNN replaces the previous bivariate linear fit so the model can
    locally adapt to curvature without needing hardcoded overrides
    for regions where the linear plane was a poor fit.
    """

    def __init__(self, points: list[tuple[float, float, float]], k: int = 5):
        # points: list of (hoop_y, hoop_x, platform_y) for past makes
        self.points = points
        self.k = min(k, len(points))

    @property
    def n(self) -> int:
        return len(self.points)

    def predict(self, hoop_y: float, hoop_x: float) -> float:
        """Return predicted optimal platform_y at this hoop position."""
        distances = sorted(
            (((py - hoop_y) ** 2 + (px - hoop_x) ** 2) ** 0.5, target)
            for py, px, target in self.points
        )
        nearest = distances[: self.k]
        # Inverse-distance weighting (clamped at 1.0 to avoid blow-up
        # for exact-match hoops).
        weights = [1.0 / max(d, 1.0) for d, _ in nearest]
        total = sum(weights)
        return sum(w * t for w, (_, t) in zip(weights, nearest)) / total


def fit_target_predictor(
    conn: sqlite3.Connection,
    required_direction: str,
    min_samples: int = 4,
    k: int = 5,
) -> KnnPredictor | None:
    """Build a KnnPredictor from clean makes in the active direction.

    Each make contributes one (hoop_y, hoop_x, platform_y) point. The
    returned object's `predict(hoop_y, hoop_x)` returns the predicted
    optimal platform_y at any hoop position.

    Excludes clamped shots — those fired at the bob extreme regardless
    of nominal target_y, so they don't reflect the predictor's signal.

    Returns None when there aren't enough makes to fit yet.
    """
    rows = list(
        conn.execute(
            'SELECT hoop_y, hoop_x, platform_y FROM shots '
            'WHERE made = 1 AND clamped = 0 AND required_direction = ? '
            'AND hoop_y IS NOT NULL AND hoop_x IS NOT NULL '
            'AND platform_y IS NOT NULL',
            (required_direction,),
        )
    )
    if len(rows) < min_samples:
        return None
    points = [(float(r[0]), float(r[1]), float(r[2])) for r in rows]
    return KnnPredictor(points, k=k)
