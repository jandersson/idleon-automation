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


def test_log_throw_records_temporal_diagnostics(tmp_path):
    """match_streak_len_before_fire and prev_match_y are the per-throw
    signals we'll use to discriminate forward-release (transient match)
    from top-of-swing apex (sustained match). prev_match_y is NULL on
    the first match of a streak."""
    conn = open_db(tmp_path / "darts.db")
    # First match of a streak — no prior y available
    log_throw(conn, throw_idx=1, match_streak_len_before_fire=1, prev_match_y=None)
    # Sustained match — fire on 4th consecutive matched frame, prior y was 320
    log_throw(conn, throw_idx=2, match_streak_len_before_fire=4, prev_match_y=320)
    rows = list(conn.execute(
        "SELECT throw_idx, match_streak_len_before_fire, prev_match_y FROM throws ORDER BY throw_idx"
    ))
    conn.close()
    assert rows == [(1, 1, None), (2, 4, 320)]


def test_migrate_adds_temporal_columns_to_old_db(tmp_path):
    """An old darts.db created before the 2026-05-24 instrumentation
    must pick up the new columns via open_db's _migrate, not error out
    when log_throw passes them."""
    import sqlite3
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE throws ("
        "  id INTEGER PRIMARY KEY, throw_idx INTEGER, release_pose_x INTEGER"
        ")"
    )
    conn.execute("INSERT INTO throws (throw_idx, release_pose_x) VALUES (1, 410)")
    conn.commit()
    conn.close()

    conn = open_db(db_path)
    log_throw(conn, throw_idx=2, match_streak_len_before_fire=3, prev_match_y=310)
    rows = list(conn.execute(
        "SELECT throw_idx, match_streak_len_before_fire, prev_match_y FROM throws ORDER BY throw_idx"
    ))
    conn.close()
    assert rows == [(1, None, None), (2, 3, 310)]


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
