"""Tests for minigames.catching.catch_log — flap-log round-trip + queries (#47)."""
from minigames.catching.catch_log import (
    open_db,
    log_flap,
    log_run,
    set_outcome,
    offset_histogram,
)


def test_open_db_creates_tables(tmp_path):
    conn = open_db(tmp_path / "catching.db")
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert {"flaps", "runs"} <= tables


def test_log_flap_round_trips(tmp_path):
    conn = open_db(tmp_path / "catching.db")
    rid = log_flap(
        conn, session_started="s", attempt_idx=1, flap_idx=1,
        fly_x=120, fly_y=300, fly_vy=140, gap_top=250, gap_bottom=320,
        gap_left_x=400, gap_right_x=470,
        gap_center=285, gap_height=70, fly_offset_below_gap=-20,
        gap_lower_margin=8, window_w=1280, window_h=720,
        code_commit="abc", source="bot",
    )
    row = conn.execute(
        "SELECT flap_idx, fly_y, fly_vy, gap_bottom, gap_left_x, gap_right_x, "
        "fly_offset_below_gap, source FROM flaps WHERE id=?", (rid,)).fetchone()
    conn.close()
    assert row == (1, 300, 140, 320, 400, 470, -20, "bot")


def test_where_keyword_column_round_trips(tmp_path):
    """`where` is a late column whose name is a SQL keyword. The migration
    must quote the identifier in the ALTER (db_log) — an unquoted ALTER raises
    a *syntax* error that the duplicate-column except swallows, so the column
    silently never gets added and the flap branch tag is lost (#60)."""
    conn = open_db(tmp_path / "catching.db")
    cols = {r[1] for r in conn.execute("PRAGMA table_info(flaps)")}
    assert "where" in cols
    rid = log_flap(conn, session_started="s", flap_idx=1, where="timed",
                   source="bot")
    val = conn.execute(
        'SELECT "where" FROM flaps WHERE id=?', (rid,)).fetchone()[0]
    conn.close()
    assert val == "timed"


def test_set_outcome_backfills(tmp_path):
    """outcome is NULL at insert (no game-over detection yet) and
    set_outcome fills it in — the scaffolding for the #47 follow-up."""
    conn = open_db(tmp_path / "catching.db")
    rid = log_flap(conn, session_started="s", flap_idx=1, source="bot")
    assert conn.execute(
        "SELECT outcome FROM flaps WHERE id=?", (rid,)).fetchone()[0] is None
    set_outcome(conn, rid, "died", measured_ms=1500)
    row = conn.execute(
        "SELECT outcome, outcome_measured_ms FROM flaps WHERE id=?", (rid,)).fetchone()
    conn.close()
    assert row == ("died", 1500)


def test_offset_histogram_bins_and_filters_source(tmp_path):
    conn = open_db(tmp_path / "catching.db")
    # bot flaps at offsets 0, 3 (bin 0) and 7 (bin 5)
    for off in (0, 3, 7):
        log_flap(conn, session_started="s", flap_idx=1,
                 fly_offset_below_gap=off, source="bot")
    # a human watch flap that the default source filter must exclude
    log_flap(conn, session_started="s", flap_idx=1,
             fly_offset_below_gap=100, source="human")
    assert offset_histogram(conn, source="bot", bin_px=5) == [(0, 2), (5, 1)]
    # source=None includes the human row too
    assert offset_histogram(conn, source=None, bin_px=5) == [(0, 2), (5, 1), (100, 1)]
    conn.close()


def test_log_run_round_trips(tmp_path):
    conn = open_db(tmp_path / "catching.db")
    rid = log_run(conn, session_started="s", attempt_idx=1,
                  n_flaps=42, duration_s=12.3, end_reason="process_exit")
    row = conn.execute(
        "SELECT n_flaps, duration_s, end_reason FROM runs WHERE id=?", (rid,)).fetchone()
    conn.close()
    assert row == (42, 12.3, "process_exit")
