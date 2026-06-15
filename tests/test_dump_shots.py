"""Smoke test for scripts/dump_shots.py — runs it against a tiny synthetic DB
in a temp dir and verifies the output JSON shape."""
import json
import sqlite3
import subprocess
import sys
from pathlib import Path


def test_dump_shots_writes_snapshot(tmp_path, monkeypatch):
    repo_root = Path(__file__).parent.parent
    fake_db = tmp_path / "shots.db"
    fake_out = tmp_path / "shots_snapshot.json"

    # Minimal schema + a few rows
    conn = sqlite3.connect(str(fake_db))
    conn.execute(
        'CREATE TABLE shots ('
        '  id INTEGER PRIMARY KEY, session_started TEXT, hoop_x INTEGER, '
        '  hoop_y INTEGER, platform_y INTEGER, "offset" INTEGER, target_y INTEGER, '
        '  clamped INTEGER, direction TEXT, required_direction TEXT, made INTEGER, '
        '  predictor_kind TEXT'
        ')'
    )
    # Three shots in the same bucket: a make_prob make, a make_prob miss, and
    # an old trajectory_gp miss. The all-data make rate is 1/3, but the
    # current-bot (make_prob) rate is 1/2.
    conn.execute(
        'INSERT INTO shots (session_started, hoop_x, hoop_y, platform_y, "offset", target_y, clamped, direction, required_direction, made, predictor_kind) '
        'VALUES (?,?,?,?,?,?,?,?,?,?,?), (?,?,?,?,?,?,?,?,?,?,?), (?,?,?,?,?,?,?,?,?,?,?)',
        (
            "2026-05-02T10:00:00", 700, 450, 470, 20, 480, 0, "up", "up", 1, "make_prob",
            "2026-05-02T10:00:00", 700, 450, 472, 20, 480, 0, "up", "up", 0, "make_prob",
            "2026-05-02T10:00:00", 700, 450, 471, 20, 480, 0, "up", "up", 0, "trajectory_gp",
        ),
    )
    conn.commit()
    conn.close()

    # Run the script with overridden paths via env-style monkeypatching of module attrs
    sys.path.insert(0, str(repo_root))
    from scripts import dump_shots
    monkeypatch.setattr(dump_shots, "DB_PATH", fake_db)
    monkeypatch.setattr(dump_shots, "OUT_PATH", fake_out)
    dump_shots.main()

    snap = json.loads(fake_out.read_text())
    assert snap["total_shots"] == 3
    assert snap["total_makes"] == 1
    assert len(snap["makes"]) == 1
    assert snap["makes"][0]["hoop_x"] == 700
    # Bucket aggregation, with recency (#43): the bucket carries the most
    # recent session that fired into it, so a reviewer can tell a stale
    # 0-make bucket from an actively-failing one.
    bucket = next(b for b in snap["buckets"]
                  if b["hoop_x_bucket"] == 700 and b["hoop_y_bucket"] == 450)
    # All-data counts blend predictors (1/3); mp_* isolate the current
    # make_prob default (1/2), so a bucket isn't misread as pesky when the
    # live bot does fine there.
    assert bucket["shots"] == 3 and bucket["makes"] == 1
    assert bucket["mp_shots"] == 2 and bucket["mp_makes"] == 1
    assert bucket["last_session"] == "2026-05-02T10:00:00"
