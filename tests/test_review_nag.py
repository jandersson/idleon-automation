"""Tests for common.review_nag — runs against a temp git repo + sqlite db."""
import sqlite3
import subprocess
from pathlib import Path

from common.review_nag import sessions_since_last_code_commit


def _git(cmd: list[str], cwd: Path) -> None:
    subprocess.run(["git", *cmd], cwd=cwd, check=True, capture_output=True)


def _init_repo(path: Path) -> None:
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "test@example.com"], path)
    _git(["config", "user.name", "Test"], path)
    _git(["config", "commit.gpgsign", "false"], path)


def _make_shot_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE shots ("
        " id INTEGER PRIMARY KEY, session_started TEXT)"
    )
    conn.commit()
    return conn


def _add_session(conn: sqlite3.Connection, iso_ts: str) -> None:
    conn.execute("INSERT INTO shots (session_started) VALUES (?)", (iso_ts,))
    conn.commit()


def test_zero_when_no_db(tmp_path):
    _init_repo(tmp_path)
    n = sessions_since_last_code_commit(tmp_path, tmp_path / "missing.db", "snap.json")
    assert n == 0


def test_counts_only_post_review_sessions(tmp_path):
    _init_repo(tmp_path)

    # Initial code commit at "now" (real time, but we don't depend on the
    # absolute time — we depend on ordering).
    (tmp_path / "main.py").write_text("# v1\n")
    _git(["add", "main.py"], tmp_path)
    _git(["commit", "-qm", "v1 code"], tmp_path)

    # Build a shot DB with sessions BEFORE and AFTER the commit.
    db = tmp_path / "shots.db"
    conn = _make_shot_db(db)
    _add_session(conn, "2020-01-01T00:00:00")  # ancient → should count as before
    _add_session(conn, "9999-01-01T00:00:00")  # future → should count as after
    _add_session(conn, "9999-01-02T00:00:00")  # future → should count as after
    conn.close()

    n = sessions_since_last_code_commit(tmp_path, db, "snap.json")
    assert n == 2  # only the two post-commit sessions


def test_snapshot_only_commits_dont_count_as_review(tmp_path):
    _init_repo(tmp_path)

    (tmp_path / "main.py").write_text("# v1\n")
    (tmp_path / "snap.json").write_text("[]")
    _git(["add", "."], tmp_path)
    _git(["commit", "-qm", "initial code+snap"], tmp_path)

    db = tmp_path / "shots.db"
    conn = _make_shot_db(db)
    _add_session(conn, "9999-01-01T00:00:00")
    conn.close()

    # A snapshot-only commit AFTER the code commit
    (tmp_path / "snap.json").write_text("[1]")
    _git(["add", "snap.json"], tmp_path)
    _git(["commit", "-qm", "snapshot refresh"], tmp_path)

    n = sessions_since_last_code_commit(tmp_path, db, "snap.json")
    # The snapshot commit should be ignored; the code commit is still the
    # latest "review" so the future session counts as post-review.
    assert n == 1
