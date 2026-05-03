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


def test_log_shot_records_click_coords(tmp_path):
    conn = open_db(tmp_path / "shots.db")
    log_shot(conn, shot_idx=1, click_x=617, click_y=372)
    row = conn.execute("SELECT click_x, click_y FROM shots").fetchone()
    conn.close()
    assert row == (617, 372)


def test_open_db_migrates_existing_db_without_click_columns(tmp_path):
    """An old DB created before click_x/click_y existed should pick them up
    via _migrate when reopened, not error out."""
    db_path = tmp_path / "old.db"
    # Create an OLD-style table missing the new columns.
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        'CREATE TABLE shots ('
        '  id INTEGER PRIMARY KEY, session_started TEXT, shot_idx INTEGER, '
        '  hoop_x INTEGER, "offset" INTEGER, made INTEGER'
        ')'
    )
    conn.execute("INSERT INTO shots (shot_idx, hoop_x) VALUES (1, 700)")
    conn.commit()
    conn.close()

    # Now reopen via open_db — should ALTER TABLE to add the new columns.
    conn = open_db(db_path)
    log_shot(conn, shot_idx=2, click_x=617, click_y=372, perturbation=-16)
    rows = list(conn.execute(
        "SELECT shot_idx, click_x, click_y, perturbation FROM shots ORDER BY id"
    ))
    conn.close()
    assert rows == [(1, None, None, None), (2, 617, 372, -16)]


def test_log_shot_records_perturbation(tmp_path):
    conn = open_db(tmp_path / "shots.db")
    log_shot(conn, shot_idx=1, perturbation=0)
    log_shot(conn, shot_idx=2, perturbation=-8)
    log_shot(conn, shot_idx=3, perturbation=24)
    rows = list(conn.execute("SELECT perturbation FROM shots ORDER BY id"))
    conn.close()
    assert rows == [(0,), (-8,), (24,)]


def test_log_shot_records_lives_diff(tmp_path):
    conn = open_db(tmp_path / "shots.db")
    log_shot(conn, shot_idx=1, lives_diff=0.0)        # no tick
    log_shot(conn, shot_idx=2, lives_diff=222.4)      # tick down
    log_shot(conn, shot_idx=3, lives_diff=None)       # region invisible
    rows = list(conn.execute("SELECT lives_diff FROM shots ORDER BY id"))
    conn.close()
    assert rows == [(0.0,), (222.4,), (None,)]


def test_log_shot_records_window_dims(tmp_path):
    conn = open_db(tmp_path / "shots.db")
    log_shot(conn, shot_idx=1, window_w=960, window_h=572)
    log_shot(conn, shot_idx=2, window_w=1280, window_h=720)  # user resized
    rows = list(conn.execute("SELECT window_w, window_h FROM shots ORDER BY id"))
    conn.close()
    assert rows == [(960, 572), (1280, 720)]


def test_fit_target_predictor_returns_none_with_too_few_samples(tmp_path):
    conn = open_db(tmp_path / "shots.db")
    for i in range(3):  # min_samples is 4 for bivariate
        log_shot(conn, hoop_y=400 + i, hoop_x=700, platform_y=420, made=1,
                 clamped=0, required_direction="up")
    assert fit_target_predictor(conn, "up") is None
    conn.close()


def test_fit_target_predictor_fits_bivariate(tmp_path):
    """Recovers a known bivariate function: platform_y = 0.5*hoop_y + 0.3*hoop_x + 10."""
    conn = open_db(tmp_path / "shots.db")
    points = [
        (300, 600), (300, 700), (400, 600), (400, 700), (500, 700), (500, 800)
    ]
    for hy, hx in points:
        py = 0.5 * hy + 0.3 * hx + 10
        log_shot(conn, hoop_y=hy, hoop_x=hx, platform_y=py, made=1,
                 clamped=0, required_direction="up")
    fit = fit_target_predictor(conn, "up")
    assert fit is not None
    a, b, c, n = fit
    assert n == 6
    assert abs(a - 0.5) < 1e-6
    assert abs(b - 0.3) < 1e-6
    assert abs(c - 10.0) < 1e-6


def test_fit_target_predictor_excludes_clamped_and_misses(tmp_path):
    conn = open_db(tmp_path / "shots.db")
    # Non-collinear points so the 3-param fit is well-conditioned.
    points = [(300, 600), (300, 700), (400, 600), (400, 700)]
    for hy, hx in points:
        py = 0.5 * hy + 0.3 * hx + 10
        log_shot(conn, hoop_y=hy, hoop_x=hx, platform_y=py, made=1,
                 clamped=0, required_direction="up")
    # Pollution that should NOT be picked up
    log_shot(conn, hoop_y=600, hoop_x=700, platform_y=999, made=1,
             clamped=1, required_direction="up")  # clamped
    log_shot(conn, hoop_y=350, hoop_x=700, platform_y=999, made=0,
             clamped=0, required_direction="up")  # miss
    log_shot(conn, hoop_y=350, hoop_x=700, platform_y=999, made=1,
             clamped=0, required_direction="down")  # wrong direction
    fit = fit_target_predictor(conn, "up")
    assert fit is not None
    a, b, c, n = fit
    assert n == 4  # only the four clean makes
    assert abs(a - 0.5) < 1e-6
    assert abs(b - 0.3) < 1e-6
