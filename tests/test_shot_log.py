"""Round-trip test for the shot SQLite log."""
import sqlite3

from common.shot_log import open_db, log_shot, fit_target_predictor


def test_open_db_creates_schema(tmp_path):
    conn = open_db(tmp_path / "shots.db")
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    conn.close()
    assert "shots" in tables


def test_log_shot_inserts_partial_row(tmp_path):
    conn = open_db(tmp_path / "shots.db")
    log_shot(
        conn,
        session_started="2026-05-01T16:00:00",
        shot_idx=1,
        hoop_x=710,
        hoop_y=448,
        offset=14,
        target_y=462,
        made=0,
        score_diff=0.0,
    )
    rows = conn.execute("SELECT shot_idx, hoop_y, \"offset\", made FROM shots").fetchall()
    conn.close()
    assert rows == [(1, 448, 14, 0)]


def test_log_shot_handles_offset_keyword(tmp_path):
    """`offset` is a SQLite reserved-ish word — make sure quoting works."""
    conn = open_db(tmp_path / "shots.db")
    log_shot(conn, shot_idx=1, offset=42)
    val = conn.execute("SELECT \"offset\" FROM shots").fetchone()[0]
    conn.close()
    assert val == 42


def test_fit_target_predictor_returns_none_with_too_few_samples(tmp_path):
    conn = open_db(tmp_path / "shots.db")
    log_shot(conn, hoop_y=400, platform_y=420, made=1, clamped=0, required_direction="up")
    log_shot(conn, hoop_y=450, platform_y=470, made=1, clamped=0, required_direction="up")
    assert fit_target_predictor(conn, "up") is None  # default min_samples=3
    conn.close()


def test_fit_target_predictor_fits_makes(tmp_path):
    """Fit uses platform_y of makes (where the platform actually was at fire)."""
    conn = open_db(tmp_path / "shots.db")
    # platform_y = 1.0 * hoop_y + 20 (perfect fit)
    for hy, py in [(300, 320), (400, 420), (500, 520)]:
        log_shot(conn, hoop_y=hy, platform_y=py, made=1, clamped=0, required_direction="up")
    fit = fit_target_predictor(conn, "up")
    assert fit is not None
    m, b, n = fit
    assert n == 3
    assert abs(m - 1.0) < 1e-9
    assert abs(b - 20.0) < 1e-9


def test_fit_target_predictor_excludes_clamped_and_misses(tmp_path):
    conn = open_db(tmp_path / "shots.db")
    # Clean makes
    for hy, py in [(300, 320), (400, 420), (500, 520)]:
        log_shot(conn, hoop_y=hy, platform_y=py, made=1, clamped=0, required_direction="up")
    # Clamped make — should be excluded (otherwise would skew the fit)
    log_shot(conn, hoop_y=600, platform_y=999, made=1, clamped=1, required_direction="up")
    # Miss — should be excluded
    log_shot(conn, hoop_y=350, platform_y=999, made=0, clamped=0, required_direction="up")
    # Wrong direction — should be excluded
    log_shot(conn, hoop_y=350, platform_y=999, made=1, clamped=0, required_direction="down")
    fit = fit_target_predictor(conn, "up")
    assert fit is not None
    m, b, n = fit
    assert n == 3  # only the three clean makes
    assert abs(m - 1.0) < 1e-9
    assert abs(b - 20.0) < 1e-9
