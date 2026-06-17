"""Tests for the fishing cast log (fish_log.py).

Mirrors test_catch_log.py / test_darts_shot_log.py: a tmp DB, the late-column
migration, a log_cast/set_outcome round-trip, and — the load-bearing bit for
the charge switch — fetch_cast_samples' NULL + source filtering, which is what
keeps pre-charge hold-era rows out of the charge<->distance fit.
"""
from minigames.fishing.fish_log import (
    open_db, log_cast, set_outcome, log_run, fetch_cast_samples,
)


def _cols(conn, table):
    return [c[1] for c in conn.execute(f"PRAGMA table_info({table})")]


def test_open_db_creates_tables_and_migrates_late_columns(tmp_path):
    db = open_db(tmp_path / "fishing.db")
    casts = _cols(db, "casts")
    runs = _cols(db, "runs")
    # late columns ALTERed in on top of the base CREATE
    assert "charge_level" in casts
    assert "target_charge_level" in casts
    assert "n_not_ready" in runs
    db.close()


def test_log_cast_set_outcome_round_trip(tmp_path):
    db = open_db(tmp_path / "fishing.db")
    rid = log_cast(db, session_started="s", cast_idx=1, hold_ms=None,
                   aim_mode="model", target_charge_level=44, source="bot")
    set_outcome(db, rid, charge_level=44, landed_dist_px=187.0,
                landed_kind="green", points=1, made=1)
    row = db.execute(
        "SELECT charge_level, landed_dist_px, target_charge_level, made "
        "FROM casts WHERE id=?", (rid,)).fetchone()
    assert row == (44, 187.0, 44, 1)
    db.close()


def _row(db, **fields):
    fields.setdefault("session_started", "s")
    log_cast(db, **fields)


def test_fetch_cast_samples_filters_null_and_source(tmp_path):
    db = open_db(tmp_path / "fishing.db")
    # (a) clean charged+landed bot cast -> always returned
    _row(db, cast_idx=1, charge_level=44, landed_dist_px=187.0, source="bot")
    # (b) legacy hold-era row: charge_level NULL -> excluded from the charge fit
    _row(db, cast_idx=2, charge_level=None, landed_dist_px=160.0, source="bot")
    # (c) charged but no landing measured -> excluded (no training y)
    _row(db, cast_idx=3, charge_level=50, landed_dist_px=None, source="bot")
    # (d) a human-watch cast with both -> returned under source=None, not 'bot'
    _row(db, cast_idx=4, charge_level=33, landed_dist_px=140.0, source="human")

    all_rows = fetch_cast_samples(db)
    assert sorted(all_rows) == [(33, 140.0), (44, 187.0)]   # a + d, b/c filtered

    bot_rows = fetch_cast_samples(db, source="bot")
    assert bot_rows == [(44, 187.0)]                         # d excluded by source
    db.close()


def test_log_run_summary(tmp_path):
    db = open_db(tmp_path / "fishing.db")
    log_run(db, session_started="s", n_casts=12, n_not_ready=4,
            points_total=7, max_streak=3, end_reason="game_over")
    row = db.execute(
        "SELECT n_casts, n_not_ready, points_total FROM runs").fetchone()
    assert row == (12, 4, 7)
    db.close()
