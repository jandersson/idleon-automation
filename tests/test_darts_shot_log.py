"""Tests for minigames.darts.shot_log — round-trip + fetch behavior."""
from minigames.darts.shot_log import open_db, log_throw, fetch_hits


def test_open_db_creates_throws_table(tmp_path):
    conn = open_db(tmp_path / "darts.db")
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    conn.close()
    assert "throws" in tables


def test_log_throw_round_trips_partial_row(tmp_path):
    conn = open_db(tmp_path / "darts.db")
    log_throw(
        conn,
        session_started="2026-05-05T01:00:00",
        throw_idx=7,
        release_pose_x=410,
        release_pose_y=250,
        release_conf=0.82,
        launch_angle_deg=-12.5,
        apex_y=110,
        landing_x=730,
        frames_seen=24,
        score_increment=2,
        hit=1,
        bullseye=0,
        streak=3,
    )
    rows = list(conn.execute(
        "SELECT throw_idx, launch_angle_deg, landing_x, score_increment, "
        "hit, bullseye, streak FROM throws"
    ))
    conn.close()
    assert rows == [(7, -12.5, 730, 2, 1, 0, 3)]


def test_fetch_hits_only_with_trajectory_excludes_null_rows(tmp_path):
    """fetch_hits with only_with_trajectory=True (default) requires
    launch_angle and landing_x both present — rows without trajectory
    data don't count as training material."""
    conn = open_db(tmp_path / "darts.db")
    log_throw(conn, throw_idx=1, launch_angle_deg=-10.0, landing_x=720, score_increment=1)
    log_throw(conn, throw_idx=2, launch_angle_deg=None, landing_x=720, score_increment=1)  # no angle
    log_throw(conn, throw_idx=3, launch_angle_deg=-10.0, landing_x=None, score_increment=1)  # no landing
    log_throw(conn, throw_idx=4, launch_angle_deg=-15.0, landing_x=600, score_increment=0)  # miss with traj
    rows = fetch_hits(conn)
    conn.close()
    # 1 and 4 survive (both have trajectory data, hit-or-miss agnostic).
    assert sorted(rows) == [(-15.0, 600, 0), (-10.0, 720, 1)]


def test_fetch_hits_includes_all_when_only_with_trajectory_false(tmp_path):
    conn = open_db(tmp_path / "darts.db")
    log_throw(conn, throw_idx=1, launch_angle_deg=-10.0, landing_x=720, score_increment=1)
    log_throw(conn, throw_idx=2, launch_angle_deg=None, landing_x=None, score_increment=0)
    rows = fetch_hits(conn, only_with_trajectory=False)
    conn.close()
    assert len(rows) == 2


def test_log_throw_handles_bullseye_flag(tmp_path):
    conn = open_db(tmp_path / "darts.db")
    log_throw(conn, throw_idx=1, score_increment=5, bullseye=1)  # bullseye
    log_throw(conn, throw_idx=2, score_increment=2, bullseye=0)  # regular
    log_throw(conn, throw_idx=3, score_increment=None, bullseye=None)  # OCR failed
    rows = list(conn.execute(
        "SELECT throw_idx, score_increment, bullseye FROM throws ORDER BY throw_idx"
    ))
    conn.close()
    assert rows == [(1, 5, 1), (2, 2, 0), (3, None, None)]
