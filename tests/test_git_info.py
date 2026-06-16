"""Tests for common.git_info — runs against a temp git repo.

Focus: the dirty-tree fingerprint (#50). A clean tree logs the bare HEAD
sha; a dirty tree logs "<sha>-dirty.<fp>" where <fp> deterministically
hashes the uncommitted changes so the rows are self-identifying even when
HEAD is an unrelated auto-commit.
"""
import re
import subprocess
from pathlib import Path

from common.git_info import current_code_commit

CLEAN = re.compile(r"^[0-9a-f]{7,}$")
DIRTY = re.compile(r"^[0-9a-f]{7,}-dirty\.[0-9a-f]{8}$")


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=path, check=True)


def _commit_seed(path: Path) -> None:
    (path / "code.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=path, check=True)


def test_clean_tree_returns_bare_sha(tmp_path):
    _init_repo(tmp_path)
    _commit_seed(tmp_path)
    cc = current_code_commit(tmp_path)
    assert cc is not None and CLEAN.match(cc), cc


def test_dirty_tracked_edit_appends_fingerprint(tmp_path):
    _init_repo(tmp_path)
    _commit_seed(tmp_path)
    (tmp_path / "code.py").write_text("x = 2  # mid-edit\n")
    cc = current_code_commit(tmp_path)
    assert cc is not None and DIRTY.match(cc), cc


def test_untracked_file_makes_tree_dirty(tmp_path):
    _init_repo(tmp_path)
    _commit_seed(tmp_path)
    (tmp_path / "new_module.py").write_text("def f(): return 1\n")
    cc = current_code_commit(tmp_path)
    assert cc is not None and DIRTY.match(cc), cc


def test_fingerprint_is_stable_for_same_state(tmp_path):
    _init_repo(tmp_path)
    _commit_seed(tmp_path)
    (tmp_path / "code.py").write_text("x = 2\n")
    assert current_code_commit(tmp_path) == current_code_commit(tmp_path)


def test_fingerprint_differs_for_different_changes(tmp_path):
    _init_repo(tmp_path)
    _commit_seed(tmp_path)
    (tmp_path / "code.py").write_text("x = 2\n")
    cc_a = current_code_commit(tmp_path)
    (tmp_path / "code.py").write_text("x = 3\n")
    cc_b = current_code_commit(tmp_path)
    # Same HEAD (no new commit) → same base sha, different fingerprint.
    assert cc_a != cc_b
    assert cc_a.split("-dirty.")[0] == cc_b.split("-dirty.")[0]


def test_non_git_dir_returns_none(tmp_path):
    assert current_code_commit(tmp_path) is None
