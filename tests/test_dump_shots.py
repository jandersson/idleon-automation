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
        '  clamped INTEGER, direction TEXT, required_direction TEXT, made INTEGER'
        ')'
    )
    conn.execute(
        'INSERT INTO shots (session_started, hoop_x, hoop_y, platform_y, "offset", target_y, clamped, direction, required_direction, made) '
        'VALUES (?,?,?,?,?,?,?,?,?,?), (?,?,?,?,?,?,?,?,?,?)',
        (
            "2026-05-02T10:00:00", 700, 450, 470, 20, 480, 0, "up", "up", 1,
            "2026-05-02T10:00:00", 700, 450, 472, 20, 480, 0, "up", "up", 0,
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
    assert snap["total_shots"] == 2
    assert snap["total_makes"] == 1
    assert snap["make_rate"] == 0.5
    assert len(snap["makes"]) == 1
    assert snap["makes"][0]["hoop_x"] == 700
    # Bucket aggregation
    assert any(b["hoop_x_bucket"] == 700 and b["hoop_y_bucket"] == 450 for b in snap["buckets"])
