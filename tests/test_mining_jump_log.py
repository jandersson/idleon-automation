"""Tests for the mining jump log — schema + insert + outcome update +
survival-rate query."""
import tempfile
from pathlib import Path

from minigames.mining.jump_log import (
    log_jump,
    open_db,
    set_outcome,
    survival_rate_by_distance,
)


def _tmpdb() -> Path:
    d = Path(tempfile.mkdtemp())
    return d / "jumps.db"


def test_open_db_creates_table():
    conn = open_db(_tmpdb())
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='jumps'"
    )
    assert cur.fetchone() == ("jumps",)


def test_log_jump_inserts_and_returns_rowid():
    conn = open_db(_tmpdb())
    row_id = log_jump(
        conn,
        session_started="2026-05-16T12:00:00",
        attempt_idx=1,
        jump_idx=1,
        next_kind="pit",
        next_distance_px=55,
    )
    assert row_id == 1
    (kind, dist) = conn.execute(
        "SELECT next_kind, next_distance_px FROM jumps WHERE id = ?", (row_id,)
    ).fetchone()
    assert kind == "pit"
    assert dist == 55


def test_set_outcome_updates_existing_row():
    conn = open_db(_tmpdb())
    row_id = log_jump(conn, jump_idx=1, next_distance_px=60)
    set_outcome(conn, row_id, "survived", measured_ms=1700)
    outcome, ms = conn.execute(
        "SELECT outcome, outcome_measured_ms FROM jumps WHERE id = ?",
        (row_id,),
    ).fetchone()
    assert outcome == "survived"
    assert ms == 1700


def test_survival_rate_buckets_by_distance():
    conn = open_db(_tmpdb())
    # 3 jumps at pit_dist=50 (bin 50), 2 survived
    for d, o in [(50, "survived"), (50, "survived"), (50, "died")]:
        rid = log_jump(conn, next_kind="pit", next_distance_px=d)
        set_outcome(conn, rid, o, measured_ms=1500)
    # 2 jumps at pit_dist=75 (bin 70), 0 survived
    for d, o in [(75, "died"), (75, "died")]:
        rid = log_jump(conn, next_kind="pit", next_distance_px=d)
        set_outcome(conn, rid, o, measured_ms=1500)
    # 1 jump at ore that shouldn't show in pit query
    rid = log_jump(conn, next_kind="ore", next_distance_px=50)
    set_outcome(conn, rid, "survived", measured_ms=1500)

    rates = survival_rate_by_distance(conn, kind="pit")
    assert (50, 3, 2) in rates
    assert (70, 2, 0) in rates
    # Ore not included
    assert not any(b == 50 and n == 4 for b, n, _ in rates)
