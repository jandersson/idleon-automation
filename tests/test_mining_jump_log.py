"""Tests for the mining jump log — schema + insert + outcome update +
survival-rate query."""
import tempfile
from pathlib import Path

from minigames.mining.jump_log import (
    log_jump,
    log_run,
    open_db,
    set_outcome,
    survival_rate_by_distance,
)


def _tmpdb() -> Path:
    d = Path(tempfile.mkdtemp())
    return d / "mining.db"


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


def test_open_db_creates_runs_table():
    conn = open_db(_tmpdb())
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
    )
    assert cur.fetchone() == ("runs",)


def test_log_run_records_final_score_and_reason():
    conn = open_db(_tmpdb())
    rid = log_run(
        conn,
        session_started="2026-05-16T12:00:00",
        attempt_idx=2,
        ended_at="2026-05-16T12:01:30",
        final_score=9,
        end_reason="play_button",
        code_commit="abc123",
    )
    row = conn.execute(
        "SELECT attempt_idx, final_score, end_reason FROM runs WHERE id = ?",
        (rid,),
    ).fetchone()
    assert row == (2, 9, "play_button")


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
        rid = log_jump(conn, next_kind="pit", next_distance_px=d, source="bot")
        set_outcome(conn, rid, o, measured_ms=1500)
    # 2 jumps at pit_dist=75 (bin 70), 0 survived
    for d, o in [(75, "died"), (75, "died")]:
        rid = log_jump(conn, next_kind="pit", next_distance_px=d, source="bot")
        set_outcome(conn, rid, o, measured_ms=1500)
    # 1 jump at ore that shouldn't show in pit query
    rid = log_jump(conn, next_kind="ore", next_distance_px=50, source="bot")
    set_outcome(conn, rid, "survived", measured_ms=1500)

    rates = survival_rate_by_distance(conn, kind="pit")
    assert (50, 3, 2) in rates
    assert (70, 2, 0) in rates
    # Ore not included
    assert not any(b == 50 and n == 4 for b, n, _ in rates)


def test_survival_rate_filters_by_source():
    conn = open_db(_tmpdb())
    rid = log_jump(conn, next_kind="pit", next_distance_px=50, source="bot")
    set_outcome(conn, rid, "survived", measured_ms=1500)
    rid = log_jump(conn, next_kind="pit", next_distance_px=50, source="human")
    set_outcome(conn, rid, "died", measured_ms=1500)

    # Default 'bot' excludes the human row.
    assert survival_rate_by_distance(conn, kind="pit") == [(50, 1, 1)]
    assert survival_rate_by_distance(conn, kind="pit", source="human") == [(50, 1, 0)]
    # source=None includes both.
    assert survival_rate_by_distance(conn, kind="pit", source=None) == [(50, 2, 1)]


def test_survival_rate_excludes_unknown_and_pending():
    conn = open_db(_tmpdb())
    rid = log_jump(conn, next_kind="pit", next_distance_px=50, source="bot")
    set_outcome(conn, rid, "survived", measured_ms=1500)
    rid = log_jump(conn, next_kind="pit", next_distance_px=50, source="bot")
    set_outcome(conn, rid, "unknown", measured_ms=1500)   # legacy outcome
    log_jump(conn, next_kind="pit", next_distance_px=50, source="bot")  # still pending (no outcome)

    assert survival_rate_by_distance(conn, kind="pit") == [(50, 1, 1)]


def test_phase_d_columns_present_and_roundtrip():
    conn = open_db(_tmpdb())
    cols = {r[1] for r in conn.execute("PRAGMA table_info(jumps)").fetchall()}
    assert {"action", "cart_height_above_plank", "cart_pose", "score_at_click"} <= cols
    rid = log_jump(conn, jump_idx=1, next_kind="pit", next_distance_px=40,
                   source="bot", action="jump", cart_height_above_plank=58,
                   cart_pose="cart_jump", score_at_click=3)
    row = conn.execute(
        "SELECT action, cart_height_above_plank, cart_pose, score_at_click "
        "FROM jumps WHERE id = ?", (rid,)).fetchone()
    assert row == ("jump", 58, "cart_jump", 3)


def test_late_columns_migrate_on_reopen_of_old_schema():
    """An old DB without the Phase-D columns gains them when reopened (the
    ALTER migration path), without losing existing rows."""
    import sqlite3
    path = _tmpdb()
    # Build a pre-Phase-D 'jumps' table (no late columns).
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE jumps (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "session_started TEXT, jump_idx INTEGER, next_distance_px INTEGER, "
                "outcome TEXT, source TEXT)")
    raw.execute("INSERT INTO jumps (jump_idx, next_distance_px) VALUES (1, 42)")
    raw.commit()
    raw.close()

    conn = open_db(path)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(jumps)").fetchall()}
    assert {"action", "cart_height_above_plank", "cart_pose", "score_at_click"} <= cols
    # Old row survives; new columns are NULL.
    row = conn.execute("SELECT next_distance_px, action FROM jumps WHERE id = 1").fetchone()
    assert row == (42, None)


def test_settle_pending_on_death_relabels_only_last_as_died():
    """On run end only the most recent jump is 'died'; earlier still-pending
    jumps 'survived' (the cart lived to make the later jump)."""
    from minigames.mining.main import _settle_pending_on_death  # lazy: avoids pyautogui import in the rest of the suite
    conn = open_db(_tmpdb())
    rids = [log_jump(conn, jump_idx=i, next_kind="pit", next_distance_px=40, source="bot")
            for i in range(1, 4)]
    pending = [{"row_id": r, "click_time": 100.0} for r in rids]
    _settle_pending_on_death(conn, pending, now=101.5)
    outs = [conn.execute("SELECT outcome FROM jumps WHERE id = ?", (r,)).fetchone()[0]
            for r in rids]
    assert outs == ["survived", "survived", "died"]
    # Single pending -> just 'died'.
    conn2 = open_db(_tmpdb())
    r = log_jump(conn2, jump_idx=1, next_kind="pit", next_distance_px=40, source="bot")
    _settle_pending_on_death(conn2, [{"row_id": r, "click_time": 0.0}], now=1.5)
    assert conn2.execute("SELECT outcome FROM jumps WHERE id = ?", (r,)).fetchone()[0] == "died"


def test_settle_outcomes_tolerates_transient_cart_detection_gaps():
    """Regression for row 59 (2026-07-06 15:07): the cart was undetected for
    a single frame at the settle instant while alive mid-rebound, and the
    outcome logged 'died'. An absent cart must keep the entry pending until
    the absence outlasts CART_GONE_CONFIRM_S; a real death (long absence)
    still settles 'died'."""
    from minigames.mining.main import _settle_outcomes

    conn = open_db(_tmpdb())
    row = log_jump(conn, session_started="s", attempt_idx=1, jump_idx=1)
    pending = [{"row_id": row, "click_time": 100.0}]

    # Settle window expired (now=102), cart undetected THIS frame but seen
    # 0.1s ago -> ambiguous, stays pending, nothing written.
    pending = _settle_outcomes(conn, pending, 102.0, None,
                               last_cart_seen_at=101.9)
    assert len(pending) == 1
    assert conn.execute("SELECT outcome FROM jumps WHERE id=?",
                        (row,)).fetchone()[0] is None

    # Cart re-detected -> survived.
    pending = _settle_outcomes(conn, pending, 102.1, (467, 179),
                               last_cart_seen_at=102.1)
    assert pending == []
    assert conn.execute("SELECT outcome FROM jumps WHERE id=?",
                        (row,)).fetchone()[0] == "survived"

    # A real death: cart unseen for longer than the confirm window.
    row2 = log_jump(conn, session_started="s", attempt_idx=1, jump_idx=2)
    pending = [{"row_id": row2, "click_time": 100.0}]
    pending = _settle_outcomes(conn, pending, 102.5, None,
                               last_cart_seen_at=101.5)
    assert pending == []
    assert conn.execute("SELECT outcome FROM jumps WHERE id=?",
                        (row2,)).fetchone()[0] == "died"
