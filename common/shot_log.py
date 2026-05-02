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
    min_samples: int = 4,
) -> tuple[float, float, float, int] | None:
    """Bivariate regression `optimal_platform_y = a*hoop_y + b*hoop_x + c`.

    Fits on the *observed* platform_y at fire time (what actually launched
    successful shots). hoop_x matters because hoops closer to the player
    need less horizontal range — pure hoop_y models extrapolate badly when
    a near-spawn happens.

    Excludes clamped shots.

    Returns (a, b, c, n) where target_y = a*hoop_y + b*hoop_x + c, or None
    when there isn't enough data.
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
    n = len(rows)
    if n < min_samples:
        return None

    # Closed-form OLS for y = a*x1 + b*x2 + c via normal equations
    # [Σx1²  Σx1x2  Σx1] [a]   [Σx1y]
    # [Σx1x2 Σx2²   Σx2] [b] = [Σx2y]
    # [Σx1   Σx2    n  ] [c]   [Σy  ]
    sx1 = sum(r[0] for r in rows)
    sx2 = sum(r[1] for r in rows)
    sy = sum(r[2] for r in rows)
    sx1x1 = sum(r[0] * r[0] for r in rows)
    sx2x2 = sum(r[1] * r[1] for r in rows)
    sx1x2 = sum(r[0] * r[1] for r in rows)
    sx1y = sum(r[0] * r[2] for r in rows)
    sx2y = sum(r[1] * r[2] for r in rows)

    M = [
        [sx1x1, sx1x2, sx1],
        [sx1x2, sx2x2, sx2],
        [sx1, sx2, n],
    ]
    v = [sx1y, sx2y, sy]
    sol = _solve_3x3(M, v)
    if sol is None:
        return None
    a, b, c = sol
    return a, b, c, n


def _solve_3x3(M: list[list[float]], v: list[float]) -> tuple[float, float, float] | None:
    """Solve a 3x3 linear system via Cramer's rule. Returns None if singular."""
    def det3(m: list[list[float]]) -> float:
        return (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )

    D = det3(M)
    if abs(D) < 1e-9:
        return None
    out = []
    for i in range(3):
        Mi = [row[:] for row in M]
        for r in range(3):
            Mi[r][i] = v[r]
        out.append(det3(Mi) / D)
    return out[0], out[1], out[2]
