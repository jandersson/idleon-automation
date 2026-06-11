"""Tests for common.auto_commit — runs against a temp git repo."""
import subprocess
from pathlib import Path

from common.auto_commit import commit_file_if_changed, _git_has_changes


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    # Disable any global hooks/signing for the test
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=path, check=True)


def test_no_commit_when_file_unchanged(tmp_path):
    _init_repo(tmp_path)
    f = tmp_path / "data.json"
    f.write_text('{"a": 1}')
    subprocess.run(["git", "add", "data.json"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)

    head_before = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True
    ).stdout.strip()

    commit_file_if_changed(tmp_path, "data.json", "shouldn't fire", push=False)

    head_after = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True
    ).stdout.strip()
    assert head_before == head_after  # no new commit


def test_commits_when_file_changed(tmp_path):
    _init_repo(tmp_path)
    f = tmp_path / "data.json"
    f.write_text('{"a": 1}')
    subprocess.run(["git", "add", "data.json"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)

    f.write_text('{"a": 2}')
    assert _git_has_changes(tmp_path, "data.json")

    commit_file_if_changed(tmp_path, "data.json", "auto: refresh data.json", push=False)

    msg = subprocess.run(
        ["git", "log", "-1", "--format=%s"], cwd=tmp_path, capture_output=True, text=True
    ).stdout.strip()
    assert msg == "auto: refresh data.json"


def test_does_not_touch_other_changes(tmp_path):
    """Auto-commit must only stage the named file, leaving unrelated edits alone."""
    _init_repo(tmp_path)
    snapshot = tmp_path / "data.json"
    other = tmp_path / "other.txt"
    snapshot.write_text('{"a": 1}')
    other.write_text("v1")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)

    snapshot.write_text('{"a": 2}')
    other.write_text("user is mid-edit, do not touch")

    commit_file_if_changed(tmp_path, "data.json", "auto", push=False)

    # The new commit should only touch data.json, NOT other.txt.
    files = subprocess.run(
        ["git", "show", "--name-only", "--format=", "HEAD"],
        cwd=tmp_path, capture_output=True, text=True,
    ).stdout.strip().splitlines()
    assert files == ["data.json"], f"new commit should only touch data.json, got: {files}"

    # And other.txt should still differ from HEAD (still has the user's edit).
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--", "other.txt"],
        cwd=tmp_path, capture_output=True, text=True,
    ).stdout
    assert "user is mid-edit" in diff


def test_commits_multiple_files_in_one_commit(tmp_path):
    """A list of paths lands as a single commit carrying every changed file."""
    _init_repo(tmp_path)
    db = tmp_path / "bot.db"
    snapshot = tmp_path / "snapshot.json"
    db.write_bytes(b"v1")
    snapshot.write_text('{"a": 1}')
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)

    db.write_bytes(b"v2")
    snapshot.write_text('{"a": 2}')

    commit_file_if_changed(tmp_path, ["bot.db", "snapshot.json"], "auto: both", push=False)

    files = subprocess.run(
        ["git", "show", "--name-only", "--format=", "HEAD"],
        cwd=tmp_path, capture_output=True, text=True,
    ).stdout.strip().splitlines()
    assert sorted(files) == ["bot.db", "snapshot.json"]
    msg = subprocess.run(
        ["git", "log", "-1", "--format=%s"], cwd=tmp_path, capture_output=True, text=True
    ).stdout.strip()
    assert msg == "auto: both"


def test_commits_untracked_file(tmp_path):
    """A brand-new (untracked) DB must still be picked up — `git diff HEAD`
    alone doesn't see untracked files."""
    _init_repo(tmp_path)
    (tmp_path / "seed.txt").write_text("x")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)

    (tmp_path / "bot.db").write_bytes(b"first session")

    commit_file_if_changed(tmp_path, "bot.db", "auto: new db", push=False)

    files = subprocess.run(
        ["git", "show", "--name-only", "--format=", "HEAD"],
        cwd=tmp_path, capture_output=True, text=True,
    ).stdout.strip().splitlines()
    assert files == ["bot.db"]
