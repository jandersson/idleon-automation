"""Tests for the chopping chop log — schema + insert + outcome update +
red-distance binning query."""
import tempfile
from pathlib import Path

from minigames.chopping.chop_log import (
    log_chop,
    log_poll,
    open_db,
    outcome_rate_by_red_distance,
    set_outcome,
)


def _tmpdb() -> Path:
    d = Path(tempfile.mkdtemp())
    return d / "chopping.db"


def test_open_db_creates_table():
    conn = open_db(_tmpdb())
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chops'"
    )
    assert cur.fetchone() == ("chops",)


def test_log_chop_inserts_and_returns_rowid():
    conn = open_db(_tmpdb())
    row_id = log_chop(
        conn,
        session_started="2026-05-24T16:00:00",
        chop_idx=1,
        pointer_x=110,
        zone="green",
        nearest_red_distance=14,
    )
    assert row_id == 1
    (zone, dist) = conn.execute(
        "SELECT zone, nearest_red_distance FROM chops WHERE id = ?", (row_id,)
    ).fetchone()
    assert zone == "green"
    assert dist == 14


def test_set_outcome_updates_existing_row():
    conn = open_db(_tmpdb())
    row_id = log_chop(conn, chop_idx=1, nearest_red_distance=12)
    set_outcome(conn, row_id, "game_over", measured_ms=480)
    outcome, ms = conn.execute(
        "SELECT outcome, outcome_measured_ms FROM chops WHERE id = ?",
        (row_id,),
    ).fetchone()
    assert outcome == "game_over"
    assert ms == 480


def test_outcome_rate_buckets_by_red_distance():
    conn = open_db(_tmpdb())
    # bin 8 (8-11px): 3 chops, 1 survived, 2 game_over
    for d, o in [(8, "survived"), (9, "game_over"), (11, "game_over")]:
        rid = log_chop(conn, nearest_red_distance=d)
        set_outcome(conn, rid, o, measured_ms=460)
    # bin 12 (12-15px): 2 chops, 2 survived
    for d, o in [(12, "survived"), (15, "survived")]:
        rid = log_chop(conn, nearest_red_distance=d)
        set_outcome(conn, rid, o, measured_ms=460)
    # Unsettled row (no red present) — excluded
    log_chop(conn, nearest_red_distance=None)
    # Settled row with NULL distance — also excluded
    rid = log_chop(conn, nearest_red_distance=None)
    set_outcome(conn, rid, "survived", measured_ms=460)

    rates = outcome_rate_by_red_distance(conn, bin_px=4)
    assert (8, 3, 1, 2) in rates
    assert (12, 2, 2, 0) in rates


def test_open_db_creates_polls_table():
    conn = open_db(_tmpdb())
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='polls'"
    )
    assert cur.fetchone() == ("polls",)


def test_log_poll_inserts_row():
    conn = open_db(_tmpdb())
    log_poll(
        conn,
        session_started="2026-05-24T17:00:00",
        t_ms=12345,
        pointer_x=110,
        zone="green",
        nearest_red_distance=14,
        bar_pixel_count=4200,
        fired=1,
    )
    conn.commit()
    row = conn.execute(
        "SELECT t_ms, pointer_x, zone, nearest_red_distance, bar_pixel_count, fired "
        "FROM polls"
    ).fetchone()
    assert row == (12345, 110, "green", 14, 4200, 1)


def test_log_poll_accepts_null_pointer_and_red_distance():
    conn = open_db(_tmpdb())
    log_poll(
        conn,
        session_started="2026-05-24T17:00:00",
        t_ms=100,
        pointer_x=None,
        zone="none",
        nearest_red_distance=None,
        bar_pixel_count=12,
        fired=0,
    )
    conn.commit()
    row = conn.execute(
        "SELECT pointer_x, nearest_red_distance, zone, bar_pixel_count "
        "FROM polls"
    ).fetchone()
    assert row == (None, None, "none", 12)
