"""Round-trip test for the shot SQLite log."""
import sqlite3

from common.shot_log import open_db, log_shot


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
